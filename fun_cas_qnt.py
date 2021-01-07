import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta
import numpy as np
import scipy.stats

def converte_index(data_inicial , num_meses):
    """
    Recebe uma data e o número de meses passados após aquela data
    Retorna uma lista com datas no formato '%Y-%m'
    """
    lista_datas = []
    for x in range(num_meses):
        lista_datas.append(data_inicial + relativedelta(months=x))
    lista_datas = pd.to_datetime(lista_datas, format='%Y-%m-%d').to_period("M")
    return lista_datas
    
def elementos_df(df):
    """
    Pega todos elementos de um DF e coloca em uma lista sem repetições
    Retorna a lista de todos elementos do DF
    """
    todos_elem = []
    for i in range(df.shape[0]):
        a = list(df.values[i])
        todos_elem += a
    todos_elem = list(set(todos_elem))
    return todos_elem

def subs_none(df):
    """
    Pega o DF e troca os não numerais por objeto None
    """
    for i in range(df.shape[0]):
        for j in range(df.shape[1]):
            if type(df.iloc[i][j]) == str:
                df.iloc[i][j] = None
    return df

def volatilidade(df):
    """
    Retorna a volatilidade a.k.a. desvio padrão população
    """
    deviations = df - df.mean()
    squared_deviations = deviations ** 2
    varianca = squared_deviations.mean()
    volatilidade = np.sqrt(varianca)
    return float(volatilidade)

def semivolatilidade(df):
    """
    Returns the semideviation aka negative deviation of r
    r must be a Series or a DataFrame
    """
    return df[df < 0].std(ddof=0)

def retorno_anual(df, periodo):
    """
    Calcula o retorno anual de um DF com retornos para a quantidade de periodos que ele da em um ano
    """
    crescimento = (1 + df).prod()
    n_periodos = df.shape[0]
    return crescimento**(periodo/n_periodos) - 1

def volatilidade_anual(df, periodo):
    """
    Calcula a volatilidade anual de um DF, com base no periodo
    """
    
    return df.std()*(periodo**0.5)

def drawdown(return_series: pd.Series):
    """
    Takes a times series of assets returns 
    Computes and returns a DataFrame that contains:
    The Wealth Index
    The Previous Peaks
    Percent Drawdowns
    """
    wealth_index = 1000*(1 + return_series).cumprod()
    previous_peaks = wealth_index.cummax()
    drawdowns = (wealth_index - previous_peaks)/previous_peaks
    return pd.DataFrame({"Montante": wealth_index, 
                        "Picos": previous_peaks, 
                         "Drawdowns": drawdowns})

def value_at_risk(df, conf=5, cornfish=False):
    """
    Returns the Parametric Gaussian VaR of a Series or DataFrame
    """
    z = scipy.stats.norm.ppf(conf/100)
    if cornfish:
        s = (((df - df.mean())**3).mean())/(df.std(ddof=0))**3
        k = (((df - df.mean())**4).mean())/(df.std(ddof=0))**4
        z = (z + 
             (z**2 - 1)*s/6 + 
             (z**3 - 3*z)*(k-3)/24 -
             (2*z**3 - 5*z)*(s**2)/36)
        
    return -(df.mean() + z*df.std(ddof=0))

def riscos_retorno(df):
    retorno = df.pct_change()
    print(f"Volatilidade: {round(volatilidade(retorno) * 100, 2)}%")
    print(f"Semivolatilidade: {round(float(semivolatilidade(retorno)) * 100, 2)}%")
    print(f"Risco Anual: {round(float(volatilidade_anual(retorno, 12)) * 100, 2)}%")
    print(f"Value at Risk (5%): {round(float(value_at_risk(retorno, cornfish=True)), 4)}")
    print(f"Montante Final: R${float(df.max())}")
    print(f"Retorno Final: {float(round(df.max()/1000, 2))}%")
    

