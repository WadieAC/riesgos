# -*- coding: utf-8 -*-
"""
Created on Tue Apr 15 13:36:54 2025

@author: n626516
"""
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np

def generar_ficheroDQ(ruta_input, ruta_salida, nombre_fichero,umbral=97.5):
    #LEEMOS FICHERO DE DQ
    ruta_DQ = ruta_input + nombre_fichero
    df_DQ =  pd.read_csv(ruta_DQ, sep=";")
    # REALIZAMOS LAS MODIFICACIONES EN EL FICHERO DE DQ
    # INICIALIZAMOS LAS COLUMNAS
    df_DQ['DQ'] = 'OK'
    df_DQ['COMENTARIO'] = ''
    # DAMOS UN KO A LOS RISK FACTORS  CON MAS DE 20 REPETIDOS CONSECUTIVOS EN AC
    df_DQ.loc[df_DQ['reps_consec'] >= 20, 'DQ'] = 'REV'
    coment = df_DQ.loc[df_DQ['reps_consec'] >= 20]['COMENTARIO']
    df_DQ.loc[df_DQ['reps_consec'] >= 20, 'COMENTARIO'] = coment + ' - El precio se repite mas de 20 veces seguidas'
    
    # DAMOS UN KO A LOS RISK FACTORS QUE TENGAN EL PORCENTAJE DE REPETIDOS DENTRO DEL PERCENTIL 97.5 EN AC
    umbral /= 100
    perc = df_DQ['%_reps'].quantile(umbral)
    df_DQ.loc[df_DQ['%_reps'] > perc, 'DQ'] = 'REV'
    coment = df_DQ.loc[df_DQ['%_reps'] > perc]['COMENTARIO']
    df_DQ.loc[df_DQ['%_reps'] > perc, 'COMENTARIO'] = coment + ' - %_reps forma parte del percentil 97.5'
    
    # DAMOS UN KO A LOS RISK FACTORS QUE TENGAN MENOS DE 512 DATOS
    df_DQ.loc[df_DQ['n_reg'] < 512, 'DQ'] = 'KO'
    coment = df_DQ.loc[df_DQ['n_reg'] < 512]['DQ']
    df_DQ.loc[df_DQ['n_reg'] < 512, 'COMENTARIO'] =  'Sin historico suficiente en VAR'
    #  DAMOS UN KO A LOS RISK FACTORS QUE TENGAN GAPS
    df_DQ.loc[df_DQ['gaps'] > 0, 'DQ'] = 'KO'
    coment = df_DQ.loc[df_DQ['gaps'] > 0]['COMENTARIO']
    df_DQ.loc[df_DQ['gaps'] > 0, 'COMENTARIO'] = coment + 'Risk factor con GAPS'
    # COGEMOS EL NOMBRE DEL FICHERO DE DQ QUE VAMOS A MODIFICAR
    ruta_DQ = ruta_salida +'Analysis_'+ nombre_fichero
    df_DQ.to_csv(ruta_DQ, sep=";", index=False)
    print(0)
    # DELVOLVEMOS LOS DATAFRAME PERIMETRO FINAL Y DF_DQ PARA POSTERIORMENTE GENERAR LOS FICHEROS QUE CONTENGAN
    # EL PERIMETRO DE TODAS LAS CURVAS Y LOS KO'S EN EL DQ DE TODAS LAS CURVAS
    return df_DQ


def generar_fichero_shock(ruta_input, ruta_salida, nombre_fichero,umbral=97.5):

    #LEEMOS FICHERO DE shock
    ruta_shock= ruta_input + nombre_fichero
    df_shock =  pd.read_csv(ruta_shock, sep=";")

    #adaptamos el umbnral
    umbral /= 100
    
    perc = df_shock['abs'].quantile(umbral)
    fichero_Shock = df_shock.copy()
    fichero_Shock['KO/OK'] = 'OK'
    fichero_Shock['abs'] = fichero_Shock['abs'].astype(float)
    fichero_Shock = fichero_Shock.sort_values('abs', ascending=False)
    fichero_Shock.loc[fichero_Shock['abs'] > perc, 'KO/OK'] = 'REV'

    ruta_shock = ruta_salida +'Analysis_'+ nombre_fichero
    fichero_Shock.to_csv(ruta_shock, sep=";", index=False)
    print(0)

    return fichero_Shock


def recta_tangente_ec(df, umbral):
    
    df_p = df.copy()
    df_p = df.loc[:,['symbol','longname','%_reps']].copy()
    #montamos una recta de 100 puntos equiespaciados a lo largo del número de muestras que tengamos
    x = np.linspace(0,100,df_p.shape[0])
    y = np.array(df.loc[:,'%_reps'])
    
    # Punto específico donde queremos calcular la tangente
    point = np.percentile(df_p.loc[:,'%_reps'], umbral) #aqui metemos el valor del percentile que queramos
    index = np.argmin(np.abs(x - point))
    
    # Calcular la derivada numérica
    dy_dx = np.gradient(y, x)
    
    # Pendiente de la tangente en el punto específico
    slope = dy_dx[index]
    y_tangent = y[index]
    x_tangent = x[index]

    print(f'Ecuación de la recta tangente en x = {x_tangent}: y = {slope} * (x - {x_tangent}) + {y_tangent}')
    # Ecuación de la recta tangente: y = m*(x - x_tangent) + y_tangent
    # y =  slope * (x - x_tangent) + y_tangent
    y_sol = slope * (0 - x_tangent) + y_tangent
    return slope,x_tangent,y_tangent,y_sol

def perfiles_repetidos(df, umbral,metrica):

    '''
    df: dataframe DQ
    umbral: 97.5
    metrica: %_reps,reps_consec, abs
    '''
        
    df_p = df.copy()
    df_p = df.loc[:,['symbol','longname',metrica]].copy()

    df_p.sort_values(by=metrica,ascending=False,inplace=True)
    df_p.index = np.arange(df_p.shape[0])
 
    #montamos una recta de 100 puntos equiespaciados a lo largo del número de muestras que tengamos
    x = np.linspace(0,df_p.shape[0],df_p.shape[0])
    y = np.array(df_p.loc[:,metrica])
    
    # Punto específico donde queremos calcular la tangente
    point = np.percentile(df_p.loc[:,metrica], umbral) #aqui metemos el valor del percentile que queramos
    index = np.argmin(np.abs(x - point))
    
    # Calcular la derivada numérica
    dy_dx = np.gradient(y, x)
    
    # Pendiente de la tangente en el punto específico
    slope = dy_dx[index]
    y_tangent = y[index]
    x_tangent = x[index]
    
    def tangent_line(x):
        return slope * (x - x_tangent) + y_tangent

    #Elimino los valores negativos que genera la derivada en la curva y acoto solo a >= 0
    vector_tangente = tangent_line(x)
    df_vector = pd.DataFrame([x,vector_tangente]).T
    df_vector.columns = ['x','tangente']
    df_vector1 = df_vector[df_vector.loc[:,'tangente']>=0]
    x_vector = np.array(df_vector1.loc[:,'x'])
    y_vector_t = np.array(df_vector1.loc[:,'tangente'])
    
    
    # Gráfica
    plt.figure(figsize=(10, 3))
    #plt.plot(x, y, label=metrica)
    plt.plot(df_p.index, y, label=metrica)
    #plt.plot(x, tangent_line(x), label='Tangente-Umbral DQ', linestyle='--')
    plt.plot(x_vector, y_vector_t, label='Tangente-Umbral DQ', linestyle='--')
    plt.scatter([x_tangent], [y_tangent], color='red', zorder=3,label='Tangent Point')
    plt.axvline(x=point, color='red',alpha=0.5,ls='--', label=f'{umbral} - Percentile')
    plt.axhline(y=point, color='red',alpha=0.8,ls='dashdot', label='Outlier Limit Zone')

    x_p = list(df_p[df_p.loc[:,metrica] >= point].index)[-1]
    label = "{:.2f}".format(point)
    plt.annotate(label, # this is the text
        (x_p,point), # these are the coordinates to position the label
        textcoords="offset points", # how to position the text
        xytext=(4,5), # distance from text to points (x,y)
        ha='left') # horizontal alignment can be left, right or center

    plt.legend()
    plt.grid()
    plt.xlabel('Risk Factor')
    plt.ylabel(f'{metrica}')
    plt.title(f'Perfil de {metrica} y umbral tangente en DQ')
    plt.show()
    
    print(f'Ecuación de la recta tangente en x = {x_tangent}: y = {slope} * (x - {x_tangent}) + {y_tangent}')

def graf_shock_curva(df,umbral):
    df2 = df.copy()
    df2.sort_values(by='abs',ascending=False,inplace=True)
    df2.index = np.arange(df2.shape[0])

    #CALCULO DE LA RECTA TANGENTE
    #montamos una recta de 100 puntos equiespaciados a lo largo del número de muestras que tengamos
    x = np.linspace(0,1000,df2.shape[0])
    y = np.array(df2.loc[:,'abs'])
    
    # Punto específico donde queremos calcular la tangente
    point = np.percentile(df2.loc[:,'abs'], umbral) #aqui metemos el valor del percentile que queramos
    index = df2[df2.loc[:,'abs'] >= point].index[-1]
    
    # Calcular la derivada numérica
    dy_dx = np.gradient(y, x)
    
    # Pendiente de la tangente en el punto específico
    slope = dy_dx[index]
    y_tangent = y[index]
    x_tangent = x[index]
    
    def tangent_line(x):
        return slope * (x - x_tangent) + y_tangent

    #Elimino los valores negativos que genera la derivada en la curva y acoto solo a >= 0
    vector_tangente = tangent_line(x)
    df_vector = pd.DataFrame([x,vector_tangente]).T
    df_vector.columns = ['x','tangente']
    df_vector1 = df_vector[df_vector.loc[:,'tangente']>=0]
    x_vector = np.array(df_vector1.loc[:,'x'])
    y_vector_t = np.array(df_vector1.loc[:,'tangente'])
    
    #Grafica de shocks Outliers
    x_p = df2[df2.loc[:,'abs'] >= point].index[-1]
    print(f'El percentil es.....{point}')
    df2.index = np.arange(df2.shape[0])
    plt.figure(figsize=(10, 3))
    plt.plot(df2.loc[:,'abs'], color='#C00000',label='Rank Shocks')
    #plt.plot(x, tangent_line(x), label='Tangente-Umbral DQ', linestyle='--')
    #plt.plot(x, vector_tangente, label='Tangente-Umbral DQ', linestyle='--')
    plt.plot(x_vector, y_vector_t, label='Tangente-Umbral DQ', linestyle='--')
    plt.scatter([x_tangent], [y_tangent], color='red', zorder=3,label='Tangent Point')
    plt.axvline(x=x_p, color='red',alpha=0.5,ls='--', label=f'{umbral} - Percentile')
    plt.axhline(y=point, color='red',alpha=0.8,ls='dashdot', label='Outlier Limit Zone')
    plt.legend()
    plt.grid()
    plt.xlabel('Risk Factor')
    plt.ylabel('Shock in Absolute value')
    plt.title('Rank shocks to the risk factor series')
    plt.tick_params(axis='x', which='both', labelbottom=False)

    label = "{:.2f}".format(point)
    plt.annotate(label, # this is the text
        (x_p,point), # these are the coordinates to position the label
        textcoords="offset points", # how to position the text
        xytext=(4,5), # distance from text to points (x,y)
        ha='left') # horizontal alignment can be left, right or center
    
    plt.tight_layout()


def plot_std_desv(df):
    df2 = df.copy()
    plt.figure(figsize=(10, 3))
    plt.bar(df2['serie'].apply(lambda x: x.split('-')[-1]), df2['std'], width=0.6, edgecolor='#FF1C1C', color='#C00000', linewidth=4)
    plt.grid(axis='y')
    plt.xlabel('Risk Factors')
    plt.title('Standard Deviation values in risk factor series')
    plt.xticks(rotation=45)
    plt.tight_layout()

def bar_plot(df,metrica):
    #df_histo.hist(bins=12, alpha=0.5, figsize=(15, 6), color='r')
    df2 = df.copy()
    inicio = df2[metrica].min()
    fin = df2[metrica].max()
    tamanio_intervalo = (fin-inicio)/5
    if tamanio_intervalo != 0:
        bins = list(np.arange(inicio, fin+0.1, tamanio_intervalo))
        labels = [f"[{round(bins[i],3)}-{round(bins[i+1]-0.001,3)}]" for i in range(len(bins) - 1)]
        df2[f'intervalos_{metrica}'] = pd.cut(df2[metrica], bins=bins, labels=labels, include_lowest=True)
        frecuencia = df2[f'intervalos_{metrica}'].value_counts().sort_index()
        plt.figure(figsize=(10, 6))
        plt.bar(frecuencia.index, frecuencia.values, width=0.6, edgecolor='#FF1C1C', color='#C00000', linewidth=4)
        plt.grid(axis='y')
        plt.title('{metrica} Distribution')
    else:
        print('Flat distribution')

def graf_perfil_metrica(df,umbral,metrica,asset,ruta_salida):
    
    df2 = df.copy()
    
    if metrica ==  'rep_consec':
        if df2.loc[:,metrica].dropna().shape[0] != df2.loc[:,metrica].shape[0]: 
            rep = df2.loc[:,'reps'].max()
            df2.loc[:,'rep_consec'] = rep
    
    df2.sort_values(by=metrica,ascending=False,inplace=True)
    df2.index = np.arange(df2.shape[0])

    #CALCULO DE LA RECTA TANGENTE
    #montamos una recta de 100 puntos equiespaciados a lo largo del número de muestras que tengamos
    x = np.linspace(0,1000,df2.shape[0])
    y = np.array(df2.loc[:,metrica])
    
    # Punto específico donde queremos calcular la tangente
    point = np.percentile(df2.loc[:,metrica], umbral) #aqui metemos el valor del percentile que queramos
    index = df2[df2.loc[:,metrica] >= point].index[-1]
    
    # Calcular la derivada numérica
    dy_dx = np.gradient(y, x)
    
    # Pendiente de la tangente en el punto específico
    slope = dy_dx[index]
    y_tangent = y[index]
    x_tangent = x[index]
    
    def tangent_line(x):
        return slope * (x - x_tangent) + y_tangent

    #Elimino los valores negativos que genera la derivada en la curva y acoto solo a >= 0
    vector_tangente = tangent_line(x)
    df_vector = pd.DataFrame([x,vector_tangente]).T
    df_vector.columns = ['x','tangente']
    max = df2.loc[:,metrica].max()
    
    df_vector1 = df_vector[(df_vector.loc[:,'tangente']>=0) & (df_vector.loc[:,'tangente']<= max)]
    
    x_vector = np.array(df_vector1.loc[:,'x'])    
    y_vector_t = np.array(df_vector1.loc[:,'tangente'])

    #Grafica de shocks Outliers
    x_p = df2[df2.loc[:,metrica] >= point].index[-1]
    print(f'El percentil para {asset} es: {point}')
    df2.index = np.arange(df2.shape[0])
    #plt.figure(figsize=(10, 3))
    fig, ax = plt.subplots(figsize=(10, 3))
    plt.plot(df2.loc[:,metrica], color='#C00000',label='Rank Values')
    plt.plot(x_vector, y_vector_t, label='Tangente-Umbral DQ', linestyle='--')
    plt.scatter([x_tangent], [y_tangent], color='red', zorder=3,alpha=0.5,label='Tangent prox Point')
    plt.scatter(x_p, point, color='red', zorder=3,label='Percentile - Point')
    plt.axvline(x=x_p, color='red',alpha=0.5,ls='--', label=f'{umbral} - Percentile')
    plt.axhline(y=point, color='red',alpha=0.8,ls='dashdot', label='Outlier Limit Zone')
    plt.legend()
    plt.grid()
    plt.xlabel('Risk Factor')
    plt.ylabel(f'{metrica} distribution values')
    plt.title(f'Rank for the risk factor series to {asset}')
    plt.tick_params(axis='x', which='both', labelbottom=False)

    label = "{:.2f}".format(point)
    plt.annotate(label, # this is the text
        (x_p,point), # these are the coordinates to position the label
        textcoords="offset points", # how to position the text
        xytext=(4,5), # distance from text to points (x,y)
        ha='left') # horizontal alignment can be left, right or center
    
    plt.tight_layout()

     
    fichero = f'{asset}__{metrica}.jpg'
    fig.savefig(ruta_salida+fichero)

    return point

def analysis_DQ(ruta_dq,ruta_salida,umbral=97.5):
    '''
    ruta_dq = r'Y:\03_DATOS_MERCADO\06_Data Quality\9. Ejercicios\2025\1Q25\DQ_summary\\'
    ruta_salida = r'Y:\03_DATOS_MERCADO\06_Data Quality\9. Ejercicios\2025\1Q25\DQ_summary\Analysis\\'
    umbral = 97.5    
    '''
    #ficheros_dq = [f for f in os.listdir(ruta_dq) if 'DQ_' in f and 'BONOS' in f ]
    ficheros_dq = [f for f in os.listdir(ruta_dq) if 'DQ_' in f]
    for dq in ficheros_dq:
        print(f'Analizando....{dq}')
        _ = generar_ficheroDQ(ruta_dq, ruta_salida,dq,umbral)

def analysis_shock(ruta_shocks,salida_shocks,umbral=97.5):
    '''
    ruta_shocks = r'Y:\03_DATOS_MERCADO\06_Data Quality\9. Ejercicios\2024\4Q24\Shock_Summary\\'
    salida_shocks = r'Y:\03_DATOS_MERCADO\06_Data Quality\9. Ejercicios\2024\4Q24\Shock_Summary\Analysis\\'
    
    '''
    ficheros_shocks = [f for f in os.listdir(ruta_shocks) if 'Shock_' in f and 'BONOS' in f ]
    for sh in ficheros_shocks:
        print(f'Analizando.....{sh}')
        _ = generar_fichero_shock(ruta_shocks, salida_shocks, sh,umbral)



def gener_perfil_rep_tabla(ruta_dq,ruta_salida,ruta_metrica,ejercicio,umbral=97.5):
    
    '''
    ruta_dq = r'Y:\03_DATOS_MERCADO\06_Data Quality\9. Ejercicios\2025\1Q25\DQ_summary\\'
    ruta_salida = r'Y:\03_DATOS_MERCADO\06_Data Quality\9. Ejercicios\2025\1Q25\DQ_summary\Analysis\plots\\'
    ruta_metrica = r'Y:\03_DATOS_MERCADO\06_Data Quality\9. Ejercicios\2025\1Q25\DQ_summary\Analysis\\'
    ejercicio = '1Q25'
    umbral = 97.5
    '''
    
    metricas = ['%_reps','reps_consec']
    ficheros_dq = [f for f in os.listdir(ruta_dq) if 'DQ_' in f]
    lista_asset = []
    lista_perc = []
    lista_metrica = []
    for sh in ficheros_dq:
        try:
            asset = sh.split('DQ_')[1].split(f'_{ejercicio}.csv')[0]
            print(f'Sacando....{asset}......')
            df_s = pd.read_csv(ruta_dq+sh,sep=';')
            for m in metricas:
                lista_asset.append(asset)
                print(f'Montando.......{m}')
                lista_metrica.append(m)
                p = graf_perfil_metrica(df_s,umbral,m,asset,ruta_salida)
                lista_perc.append(p)
        except OSError:
            pass
            
    #Generación de tabla resumen
    #---------------------------------------------------------------------
    umb_str = str(umbral)
    df_metricas = pd.DataFrame([lista_asset,lista_perc,lista_metrica]).T
    df_metricas.columns = ['Asset_Class',f'Perc_{umb_str}','metrica']
    df_metricas = pd.pivot_table(df_metricas, values=f'Perc_{umb_str}', index=['Asset_Class'],
                           columns=['metrica'])
    df_metricas.reset_index(inplace=True)
    df_metricas.to_excel(ruta_metrica+f'Metricas_reps_{ejercicio}.xlsx')

def gener_perfil_shock_tabla(ruta_shocks,ruta_salida,ruta_metrica,ejercicio,umbral=97.5):

    '''
    ruta_shocks = r'Y:\03_DATOS_MERCADO\06_Data Quality\9. Ejercicios\2024\4Q24\Shock_Summary\\'
    salida_shocks = r'Y:\03_DATOS_MERCADO\06_Data Quality\9. Ejercicios\2024\4Q24\Shock_Summary\Analysis\\'
    ruta_salida = r'Y:\03_DATOS_MERCADO\06_Data Quality\9. Ejercicios\2024\4Q24\Shock_Summary\plots\\'
    ejercicio = '1Q25'
    umbral = 97.5
    '''
    
    ficheros_shocks = [f for f in os.listdir(ruta_shocks) if 'Shock_' in f]
    metricas = [ 'abs']
    lista_asset = []
    lista_perc = []
    lista_metrica = []
    for sh in ficheros_shocks:
        asset = sh.split('Shock_')[1].split(f'_{ejercicio}.csv')[0]
        print(f'Sacando....{asset}......')
        df_s = pd.read_csv(ruta_shocks+sh,sep=';')
        for m in metricas:
            lista_asset.append(asset)
            print(f'Montando.......{m}')
            lista_metrica.append(m)
            p = graf_perfil_metrica(df_s,umbral,m,asset,ruta_salida)
            lista_perc.append(p)
    
    #Generación de tabla resumen
    #---------------------------------------------------------------------
    umb_str = str(umbral)
    df_metricas = pd.DataFrame([lista_asset,lista_perc,lista_metrica]).T
    df_metricas.columns = ['Asset_Class',f'Perc_{umb_str}','metrica']
    df_metricas = pd.pivot_table(df_metricas, values=f'Perc_{umb_str}', index=['Asset_Class'],
                           columns=['metrica'])
    df_metricas.reset_index(inplace=True)
    ruta_metrica = ruta_salida
    df_metricas.to_excel(ruta_metrica+'Metricas_Shocks_{ejercicio}.xlsx')

