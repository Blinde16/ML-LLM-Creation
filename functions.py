#Creating my own class library for automated
def univariate_stats(df, roundto=4):
  import pandas as pd
  df_results = pd.DataFrame(columns=['dtype', 'count', 'missing', 'unique', 'mode', 
                                      'min', 'q1', 'median', 'q3', 'max',
                                      'mean', 'std', 'skew', 'kurt'])
  
  for col in df.columns:
      dtype = df[col].dtype
      count = df[col].count()
      missing = df[col].isna().sum()
      unique = df[col].nunique()
      mode = df[col].mode().iloc[0] if not df[col].mode().empty else "N/A"

      if pd.api.types.is_numeric_dtype(df[col]):
          min_val = df[col].min()
          q1 = df[col].quantile(0.25)
          median = df[col].median()
          q3 = df[col].quantile(0.75)
          max_val = df[col].max()
          mean = df[col].mean()
          std = df[col].std()
          skew = df[col].skew()
          kurt = df[col].kurt()

          df_results.loc[col] = [dtype, count, missing, unique, mode, 
                                  round(min_val, roundto), round(q1, roundto), round(median, roundto), round(q3, roundto), round(max_val, roundto), 
                                  round(mean, roundto), round(std, roundto), round(skew, roundto), round(kurt, roundto)]
      else:
          df_results.loc[col] = [dtype, count, missing, unique, mode, 
                                  "", "", "", "", "", 
                                  "", "", "", ""]

  return df_results
def basic_wrangling(df, features=[], missing_threshold=0.95, unique_threshold=0.95, messages=True):
  import pandas as pd
  
  if not features: 
      features = df.columns
  
  for feat in features:
    if feat in df.columns:
      missing = df[feat].isna().sum()
      unique = df[feat].nunique()
      rows = df.shape[0]
  
      if missing / rows >= missing_threshold:
        if messages: print(f"Dropping {feat}: {missing} missing values out of {rows} ({round(missing/rows, 2)})")
        df.drop(columns=[feat], inplace=True)
      elif unique / rows >= unique_threshold:
        if df[feat].dtype in ['int64', 'object']:
          if messages: print(f"Dropping {feat}: {unique} unique values out of {rows} ({round(unique/rows, 2)})")
          df.drop(columns=[feat], inplace=True)
      elif unique == 1:
        if messages: print(f"Dropping {feat}: Contains only one unique value ({df[feat].unique()[0]})")
        df.drop(columns=[feat], inplace=True)
    else:
      if messages: print(f"Skipping \"{feat}\": Column not found in DataFrame")
  
  return df

def parse_date(df, features=[], days_since_today=False, drop_date=True, messages=True):
  import pandas as pd
  from datetime import datetime as dt
  
  for feat in features:
    if feat in df.columns:
      df[feat] = pd.to_datetime(df[feat])
      df[f'{feat}_year'] = df[feat].dt.year
      df[f'{feat}_month'] = df[feat].dt.month
      df[f'{feat}_day'] = df[feat].dt.day
      df[f'{feat}_weekday'] = df[feat].dt.day_name()
    
      if days_since_today:
        df[f'{feat}_days_until_today'] = (dt.today() - df[feat]).dt.days
      if drop_date:
        df.drop(columns=[feat], inplace=True)
    else:
      if messages:
        print(f'{feat} does not exist in the DataFrame provided. No work performed.')
  
  return df

  def univariate_charts(df, box=True, hist=True, save=False, save_path='', stats=True):
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt

    sns.set(style="ticks")

    for col in df.columns:
      plt.figure(figsize=(8, 5))

      if pd.api.types.is_numeric_dtype(df[col]):
        if box and hist:
          fig, (ax_box, ax_hist) = plt.subplots(
              2, sharex=True, gridspec_kw={"height_ratios": (0.2, 0.8)}, figsize=(8, 5)
          )
          sns.boxplot(x=df[col], ax=ax_box, fliersize=4, width=0.5, linewidth=1)
          sns.histplot(df[col], kde=True, ax=ax_hist)
          ax_box.set(yticks=[], xlabel='')
          sns.despine(ax=ax_box, left=True)
          sns.despine(ax=ax_hist)
        elif box:
          sns.boxplot(x=df[col], fliersize=4, width=0.5, linewidth=1)
          sns.despine()
        elif hist:
          sns.histplot(df[col], kde=True, rug=True)
          sns.despine()

        if stats:
          stats_text = (
            f"Unique: {df[col].nunique()}\n"
            f"Missing: {df[col].isnull().sum()}\n"
            f"Mode: {df[col].mode().iloc[0]}\n"
            f"Min: {df[col].min():.2f}\n"
            f"25%: {df[col].quantile(0.25):.2f}\n"
            f"Median: {df[col].median():.2f}\n"
            f"75%: {df[col].quantile(0.75):.2f}\n"
            f"Max: {df[col].max():.2f}\n"
            f"Std dev: {df[col].std():.2f}\n"
            f"Mean: {df[col].mean():.2f}\n"
            f"Skew: {df[col].skew():.2f}\n"
            f"Kurt: {df[col].kurt():.2f}"
          )
          plt.gcf().text(0.95, 0.5, stats_text, fontsize=10, va='center', transform=plt.gcf().transFigure)
      else:
        sns.countplot(x=col, data=df, order=df[col].value_counts().index, hue=col, dodge=False, legend=False, palette="RdBu_r")
        sns.despine()
        if stats:
          stats_text = (
            f"Unique: {df[col].nunique()}\n"
            f"Missing: {df[col].isnull().sum()}\n"
            f"Mode: {df[col].mode().iloc[0]}"
          )
          plt.gcf().text(0.95, 0.5, stats_text, fontsize=10, va='center', transform=plt.gcf().transFigure)

      plt.title(col, fontsize=14)
      if save:
        plt.savefig(f"{save_path}{col}.png", dpi=100, bbox_inches='tight')
      plt.show()

def scatterplot(df, feature, label, roundto=3, linecolor='darkorange'):
  import pandas as pd
  from matplotlib import pyplot as plt
  import seaborn as sns
  from scipy import stats

  # Create the plot
  sns.regplot(x=df[feature], y=df[label], line_kws={"color": linecolor})

  # Calculate the regression line so that we can print the text
  m, b, r, p, err = stats.linregress(df[feature], df[label])

  # Add all descriptive statistics to the diagram
  textstr  = 'Regression line:' + '\n'
  textstr += 'y  = ' + str(round(m, roundto)) + 'x + ' + str(round(b, roundto)) + '\n'
  textstr += 'r   = ' + str(round(r, roundto)) + '\n'
  textstr += 'r2 = ' + str(round(r**2, roundto)) + '\n'
  textstr += 'p  = ' + str(round(p, roundto)) + '\n\n'

  plt.text(1, 0.1, textstr, fontsize=12, transform=plt.gcf().transFigure)
  plt.show()

def bar_chart(df, feature, label, roundto=3):
  import pandas as pd
  from scipy import stats
  from matplotlib import pyplot as plt
  import seaborn as sns

  # Handle missing data
  df_temp = df[[feature, label]]
  df_temp = df_temp.dropna()

  sns.barplot(df_temp, x=feature, y=label)

  sns.barplot(df, x=feature, y=label)

  # Create the label lists needed to calculate oneway-ANOVA F
  groups = df[feature].unique()
  group_lists = []
  for g in groups:
    g_list = df[df[feature] == g][label]
    group_lists.append(g_list)

  results = stats.f_oneway(*group_lists)
  F = results[0]
  p = results[1]

  # Next, calculate t-tests with Bonferroni correction for p-value threshold
  ttests = []
  for i1, g1 in enumerate(groups): # Use the enumerate() function to add an index for counting to a list of values
    # For each item, loop through a second list of each item to compare each pair
    for i2, g2 in enumerate(groups):
      if i2 > i1: # If the inner_index is greater that the outer_index, then go ahead and run a t-test
        type_1 = df[df[feature] == g1]
        type_2 = df[df[feature] == g2]
        t, p = stats.ttest_ind(type_1[label], type_2[label])

        # Add each t-test result to a list of t, p pairs
        ttests.append([str(g1) + ' - ' + str(g2), round(t, roundto), round(p, roundto)])

  p_threshold = 0.05 / len(ttests) # Bonferroni-corrected p-value determined

  # Add all descriptive statistics to the diagram
  textstr  = '   ANOVA' + '\n'
  textstr += 'F: ' + str(round(F, roundto)) + '\n'
  textstr += 'p: ' + str(round(p, roundto)) + '\n\n'

  # Only include the significant t-tests in the printed results for brevity
  for ttest in ttests:
    if ttest[2] <= p_threshold:
      if 'Sig. comparisons (Bonferroni-corrected)' not in textstr: # Only include the header if there is at least one significant result
        textstr += 'Sig. comparisons (Bonferroni-corrected)' + '\n'
      textstr += str(ttest[0]) + ": t=" + str(ttest[1]) + ", p=" + str(ttest[2]) + '\n'

  plt.text(1, 0.1, textstr, fontsize=12, transform=plt.gcf().transFigure)
  plt.show()

def bin_categories(df, feature, cutoff=0.05, replace_with='Other'):
  # create a list of feature values that are below the cutoff percentage
  other_list = df[feature].value_counts()[df[feature].value_counts() / len(df) < cutoff].index

  # Replace the value of any country in that list (using the .isin() method) with 'Other'
  df.loc[df[feature].isin(other_list), feature] = replace_with

  return df

#Crosstab
def crosstab(df, feature, label, roundto=3):
  import pandas as pd
  from scipy.stats import chi2_contingency
  from matplotlib import pyplot as plt
  import seaborn as sns
  import numpy as np

  # Handle missing data
  df_temp = df[[feature, label]]
  df_temp = df_temp.dropna()

  # Bin categories
  df_temp = bin_categories(df_temp, feature)

  # Generate the crosstab table required for X2
  crosstab = pd.crosstab(df_temp[feature], df_temp[label])

  # Calculate X2 and p-value
  X, p, dof, contingency_table = chi2_contingency(crosstab)

  textstr  = 'X2: ' + str(round(X, 4))+ '\n'
  textstr += 'p = ' + str(round(p, 4)) + '\n'
  textstr += 'dof  = ' + str(dof)
  plt.text(0.9, 0.1, textstr, fontsize=12, transform=plt.gcf().transFigure)

  ct_df = pd.DataFrame(np.rint(contingency_table).astype('int64'), columns=crosstab.columns, index=crosstab.index)
  sns.heatmap(ct_df, annot=True, fmt='d', cmap='coolwarm')
  plt.show()
  
def bivariate(df, label, roundto=4):
  import pandas as pd
  from scipy import stats

  output_df = pd.DataFrame(columns=['missing', 'p', 'r', 'τ', 'ρ', 'y = m(x) + b', 'F', 'X2', 'skew', 'unique', 'values'])

  for feature in df.columns:
    if feature != label:
      df_temp = df[[feature, label]]
      df_temp = df_temp.dropna()
      missing = (df.shape[0] - df_temp.shape[0]) / df.shape[0]
      unique = df_temp[feature].nunique()

      # Bin categories
      if not pd.api.types.is_numeric_dtype(df_temp[feature]):
        df = bin_categories(df, feature)

      if pd.api.types.is_numeric_dtype(df_temp[feature]) and pd.api.types.is_numeric_dtype(df_temp[label]):
        m, b, r, p, err = stats.linregress(df_temp[feature], df_temp[label])
        tau, tp = stats.kendalltau(df_temp[feature], df_temp[label])
        rho, rp = stats.spearmanr(df_temp[feature], df_temp[label])
        output_df.loc[feature] = [f'{missing:.2%}', round(p, roundto), round(r, roundto), round(tau, roundto),
                                  round(rho, roundto), f'y = {round(m, roundto)}(x) + {round(b, roundto)}', '-', '-',
                                  df_temp[feature].skew(), unique, '-']

        scatterplot(df_temp, feature, label, roundto) # Call the scatterplot function
      elif not pd.api.types.is_numeric_dtype(df_temp[feature]) and not pd.api.types.is_numeric_dtype(df_temp[label]):
        contingency_table = pd.crosstab(df_temp[feature], df_temp[label])
        X2, p, dof, expected = stats.chi2_contingency(contingency_table)
        output_df.loc[feature] = [f'{missing:.2%}', round(p, roundto), '-', '-', '-', '-', '-', round(X2, roundto), '-',
                                  unique, df_temp[feature].unique()]

        crosstab(df_temp, feature, label, roundto) # Call the crosstab function
      else:
        if pd.api.types.is_numeric_dtype(df_temp[feature]):
          skew = df_temp[feature].skew()
          num = feature
          cat = label
        else:
          skew = '-'
          num = label
          cat = feature

        groups = df_temp[cat].unique()
        group_lists = []
        for g in groups:
          g_list = df_temp[df_temp[cat] == g][num]
          group_lists.append(g_list)

        results = stats.f_oneway(*group_lists)
        F = results[0]
        p = results[1]
        output_df.loc[feature] = [f'{missing:.2%}', round(p, roundto), '-', '-', '-', '-', round(F, roundto), '-', skew,
                                  unique, df_temp[cat].unique()]

        bar_chart(df_temp, cat, num, roundto) # Call the barchart function
  return output_df.sort_values(by=['p'])
