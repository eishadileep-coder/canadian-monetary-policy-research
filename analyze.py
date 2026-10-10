"""Canadian policy rates and provincial unemployment: descriptive, NOT causal.

Run with Python 3.13 using the installed requirements.txt packages.
Public sources: Bank of Canada Valet V39079; Statistics Canada table 14-10-0287-01.
"""
from pathlib import Path
from io import BytesIO
import zipfile
import requests
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

BASE = Path(__file__).resolve().parent
OUT = BASE / 'outputs'
OUT.mkdir(exist_ok=True)
HEADERS = {'User-Agent': 'University student economics research project (educational)'}
BOC = 'https://www.bankofcanada.ca/valet/observations/V39079/json?start_date=2005-01-01'
STATS = 'https://www150.statcan.gc.ca/n1/en/tbl/csv/14100287-eng.zip'
PROVINCES = ['Ontario', 'Quebec', 'Alberta', 'British Columbia']
START, END = '2005-01-01', '2026-09-01'


def fetch_policy():
    print('1/5 Downloading Bank of Canada policy rates...', flush=True)
    response = requests.get(BOC, headers=HEADERS, timeout=90)
    response.raise_for_status()
    observations = response.json()['observations']
    rows = [{'date': r['d'], 'policy_rate': float(r['V39079']['v'])}
            for r in observations if 'V39079' in r and r['V39079'].get('v')]
    if not rows:
        raise ValueError('No policy rates returned; confirm Bank of Canada V39079 series.')
    rates = pd.DataFrame(rows)
    rates['date'] = pd.to_datetime(rates['date'])
    rates = (rates.set_index('date').sort_index()
             .resample('MS').last().ffill().reset_index())
    return rates


def fetch_unemployment():
    print('2/5 Downloading Statistics Canada unemployment data (large ZIP)...', flush=True)
    response = requests.get(STATS, headers=HEADERS, timeout=180)
    response.raise_for_status()
    print('    Filtering CSV in chunks to reduce memory usage...', flush=True)
    collected = []
    with zipfile.ZipFile(BytesIO(response.content)) as archive:
        csvs = [name for name in archive.namelist()
                if name.lower().endswith('.csv') and 'metadata' not in name.lower() and 'symbol' not in name.lower()]
        if not csvs:
            raise ValueError('Statistics Canada ZIP contains no data CSV.')
        csv_name = max(csvs, key=lambda n: archive.getinfo(n).file_size)
        with archive.open(csv_name) as source:
            reader = pd.read_csv(source, chunksize=150_000, low_memory=False)
            for part in reader:
                required = {'REF_DATE', 'GEO', 'VALUE'}
                if not required.issubset(part.columns):
                    raise ValueError(f'Statistics Canada CSV missing: {required - set(part.columns)}')
                part = part[part['GEO'].isin(PROVINCES)].copy()
                if part.empty:
                    continue
                # Require an explicit unemployment-rate statistic, rather than guessing a fallback.
                candidates = [c for c in ('Labour force characteristics', 'Statistics') if c in part.columns]
                if not candidates:
                    raise ValueError('No labour-force measure column available; inspect StatsCan table schema.')
                condition = pd.Series(False, index=part.index)
                for column in candidates:
                    condition |= part[column].astype(str).str.strip().str.casefold().eq('unemployment rate')
                part = part[condition]
                for column in ('Age group',):
                    if column in part:
                        part = part[part[column].astype(str).str.strip().str.casefold().isin(
                            ('15 years and over', '15 years and older'))]
                for column in ('Gender', 'Sex'):
                    if column in part:
                        part = part[part[column].astype(str).str.strip().str.casefold().isin(
                            ('total - gender', 'both sexes', 'total - sex'))]
                # This source has multiple estimates and data types per month.
                # Explicitly select the official seasonally adjusted estimate.
                if 'Statistics' not in part.columns or 'Data type' not in part.columns:
                    raise ValueError('Expected Statistics and Data type columns are absent.')
                part = part[part['Statistics'].astype(str).str.strip().str.casefold().eq('estimate')]
                part = part[part['Data type'].astype(str).str.strip().str.casefold().eq('seasonally adjusted')]
                if not part.empty:
                    cols = ['REF_DATE', 'GEO', 'VALUE', 'Statistics', 'Data type']
                    collected.append(part[cols].copy())
    if not collected:
        raise ValueError('No rows matched seasonally adjusted unemployment rate, age 15+, total gender, estimate. Inspect source table dimensions.')
    unemployment = pd.concat(collected, ignore_index=True)
    unemployment['date'] = pd.to_datetime(unemployment['REF_DATE'], errors='coerce')
    unemployment['unemployment_rate'] = pd.to_numeric(unemployment['VALUE'], errors='coerce')
    unemployment = (unemployment.rename(columns={'GEO': 'province'})
                    [['date', 'province', 'unemployment_rate']].dropna())
    duplicates = unemployment.duplicated(['date', 'province'], keep=False)
    if duplicates.any():
        sample = unemployment.loc[duplicates].head(8).to_string(index=False)
        raise ValueError('Multiple observations remain after filtering. Inspect the source dimensions:\n' + sample)
    if len(unemployment) < 100:
        raise ValueError('Too few unemployment observations; verify published data schema.')
    return unemployment.sort_values(['province', 'date']).reset_index(drop=True)


def save_model(name, formula, data):
    # One observation per date for the national-rate specification.
    fit = smf.ols(formula, data=data).fit(cov_type='HAC', cov_kwds={'maxlags': 12})
    (OUT / f'{name}.txt').write_text(fit.summary().as_text(), encoding='utf-8')
    return fit


def main():
    rates = fetch_policy()
    unemployment = fetch_unemployment()
    rates.to_csv(OUT / 'bank_of_canada_policy_rates.csv', index=False)
    unemployment.to_csv(OUT / 'statistics_canada_unemployment.csv', index=False)
    print('3/5 Joining and validating monthly observations...', flush=True)
    panel = unemployment.merge(rates, on='date', how='inner')
    panel = panel[panel['date'].between(START, END)].copy()
    panel = panel.sort_values(['province', 'date'])
    if panel.empty:
        raise ValueError('No overlap between unemployment data and policy-rate dates.')
    # Require four provincial observations each month for consistent averages.
    counts = panel.groupby('date')['province'].nunique()
    complete_dates = counts[counts == len(PROVINCES)].index
    panel = panel[panel['date'].isin(complete_dates)].copy()
    if panel.empty:
        raise ValueError('No months have observations for all four provinces.')
    if panel.groupby(['date', 'province']).size().max() != 1:
        raise ValueError('Duplicate province-month observations remain.')
    panel.to_csv(OUT / 'provincial_analysis_dataset.csv', index=False)

    # A single monthly observation combines four equally weighted provinces.
    # This avoids false precision from repeating a shared national policy rate.
    monthly = (panel.groupby('date', as_index=False)
               .agg(unemployment_rate=('unemployment_rate', 'mean'),
                    policy_rate=('policy_rate', 'first'))
               .sort_values('date'))
    monthly['month'] = monthly['date'].dt.month.astype(str)
    monthly['year'] = monthly['date'].dt.year.astype(str)
    # Shift AFTER producing one row per month. This is a lagged association.
    for lag in (0, 3, 6, 12):
        monthly[f'policy_rate_lag{lag}'] = monthly['policy_rate'].shift(lag)
    monthly.to_csv(OUT / 'analysis_dataset.csv', index=False)

    print('4/5 Estimating alternative descriptive regressions...', flush=True)
    results = []
    for lag in (0, 3, 6, 12):
        subset = monthly.dropna(subset=[f'policy_rate_lag{lag}', 'unemployment_rate']).copy()
        if len(subset) < 50:
            raise ValueError(f'Too few observations for lag {lag}.')
        name = f'lag{lag}_month_controls'
        model = save_model(name,
                           f'unemployment_rate ~ policy_rate_lag{lag} + C(month)', subset)
        key = f'policy_rate_lag{lag}'
        results.append({'specification': name, 'lag_months': lag,
                        'coefficient': model.params[key], 'std_error_HAC': model.bse[key],
                        'p_value': model.pvalues[key], 'r_squared': model.rsquared,
                        'n_months': int(model.nobs)})
    # The original model's year controls are kept only as a sensitivity comparison.
    subset = monthly.dropna(subset=['policy_rate_lag3', 'unemployment_rate']).copy()
    year_model = save_model('lag3_month_and_year_controls',
                            'unemployment_rate ~ policy_rate_lag3 + C(month) + C(year)', subset)
    results.append({'specification': 'lag3_month_and_year_controls', 'lag_months': 3,
                    'coefficient': year_model.params['policy_rate_lag3'],
                    'std_error_HAC': year_model.bse['policy_rate_lag3'],
                    'p_value': year_model.pvalues['policy_rate_lag3'],
                    'r_squared': year_model.rsquared, 'n_months': int(year_model.nobs)})
    summary = pd.DataFrame(results)
    summary.to_csv(OUT / 'model_comparison.csv', index=False)

    print('5/5 Saving charts and research notes...', flush=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    for province in PROVINCES:
        x = panel[panel['province'] == province]
        ax.plot(x['date'], x['unemployment_rate'], label=province, linewidth=1)
    ax.set(title='Provincial monthly unemployment rates (seasonally adjusted)',
           ylabel='Unemployment rate (%)')
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / 'unemployment_trends.png', dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(monthly['date'], monthly['policy_rate'])
    ax.set(title='Bank of Canada target overnight rate', ylabel='Percent')
    fig.tight_layout()
    fig.savefig(OUT / 'policy_rate.png', dpi=160)
    plt.close(fig)

    lines = [
        '# Canadian Monetary Policy and Unemployment — Descriptive Evidence',
        '',
        '## Data and design',
        'Bank of Canada V39079 and Statistics Canada table 14-10-0287-01.',
        f'Observed window: {monthly.date.min():%Y-%m} to {monthly.date.max():%Y-%m}; {len(monthly)} months.',
        'Unemployment measure: seasonally adjusted unemployment rates for Ontario, Quebec, Alberta and British Columbia.',
        'Outcome: simple (equal-province) average; **not** the official Canada unemployment rate.',
        'Covariance: Newey-West/HAC standard errors with 12 monthly lags.',
        '',
        '## Estimated associations',
        'See model_comparison.csv and model-specific regression text files.',
        'These coefficients describe conditional associations, not causal effects.',
        'Different lag samples and controls also complicate comparisons.',
        '',
        '## Interpretation questions',
        '1. Are coefficient signs and p-values stable across 0-, 3-, 6- and 12-month lags?',
        '2. How do year controls affect the 3-month coefficient?',
        '3. What roles do inflation, COVID-era shocks, policy endogeneity, and trends play?',
        '4. What identification strategy would be needed to estimate causal monetary-policy effects?',
        '',
        '## Limitations',
        '- The sample represents four provinces and an equal-weighted average, not all of Canada.',
        '- The national policy rate is shared by provinces; province-level replication does not increase independent policy variation.',
        '- No causal identification; interest-rate decisions respond to economic conditions.',
        '- Descriptive results may be sensitive to sample choices, seasonality, serial correlation, and lags.',
    ]
    (OUT / 'research_brief.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('Finished:', OUT, flush=True)
    print('\nCompare your models in: outputs/model_comparison.csv', flush=True)
    print(summary[['specification', 'coefficient', 'p_value', 'n_months']].to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
