# Desafio Data Science - Lighthouse

## O problema

O desafio tem como objetivo avaliar o desenvolvimento de uma EDA (exploratory data analysis) e os conhecimentos/insights relacionados a um tipo clássico de problema de data science: séries temporais. Para isso, gostaríamos que fosse previsto o crescimento do índice GDP de cada país nos anos de 2024-2028, posteriormente comparando-os com o previsto pelo Statistica. Além disso, é necessário substituir os campos "no data" por valores numéricos, utilizando inferências de sua escolha.

### Informações sobre os dados:

- Os valores numéricos de GDP são valores percentuais.
- O dataset apresenta valores preenchidos como 'no data'.
- O próprio dataset ja traś valores de GDP para 2024 a 2028, resultados de uma previsão de série temporal cujo método é desconhecido.

## Primeiros passos para solução do problema

### 1) Análise do dataset

Entender como os valores 'no data' estão distribuídos para os países, substitui-los por 'None' e verificar o percentual que os valores nulos representam para cada país.

Foi verificado que o dataset também contém os valors de GPD para grupos/categorias, dividos em: economias, regiões, continentes e conjunto de continentes.

### 2) Países por grupo/categoria

Foram criadas diversas listas com os mesmos nomes dos grupos e categorias já existentes e, definido quais paises portenceriam a eles. Dessa forma, foi realizada uma análise gráfica dos dados considerados do passado (1980 até 2023) e verificado se haveria alguma semelhança nas curvas desses paises. 

A partir desse momento, foram análisados todos os paises que apresentavam dados faltantes e definido quais os grupos que seriam utilizados para susbtituição dos valores nulos. 

#### a) Os seguintes grupos foram utilizados para substituir os valores nulos dos paises:

- South Asia
- Advanced Economies
- Africa
- Europe
- Middle East
- Pacific Islands
- Southeast Asia
- Soviet Union* 
- Central America

**Este foi o único que não havia no dataset, mas foi criado para preencher os dados faltantes. A justificativa para criação desse grupo foi por meio da análise grafica dos paises que faziam parte da União Soviética, e o entendimento de que todos tendiam a ter uma economia semelhante ao período que faziam parte da mesma união.*

#### b) Os seguintes grupos foram utilizados para substituir os valores nulos dos próprios grupos:

- Sub-Saharan Africa (Region)
- Sub-Saharan Africa
- Euro area
- Middle East
- Central Asia and the Caucasus
- Central America

### 3) Definição do método para substituição dos valores nulos

#### 3.1) Preenchendo os valores do passado

Os passos tomados foram:

##### 3.1.1) Cálculo da média móvel

Foi calculada a média móvel anual com base em todos os países pertencentes ao mesmo grupo. Porém, para calcular a média, foram considerados apenas dos dados de 1980 até 2019. Esta abordagem foi implementada por causa da COVID-19, que fez ocorrer uma anomalia a nível global. 

*Vale ressaltar que os valores dos grupos/categorias **não** foram utilziados para o cálulo da média móvel.*

##### 3.1.2) Exceções para poucos valores nulos

Para casos de países que apresentam apenas 2 valores nulos em toda a sua série passada, foi apenas encontrada a média para este próprio pais e preenchido nos dados faltantes. 

##### 3.1.3) Identificando outliers

Encontrar os outliers de cada país. Foi usado a técnica no boxplot para encontrar os valores considerados outliers, usando a seguinte regra:

    q1 = np.percentile(series_sem_nan, 25)
    q3 = np.percentile(series_sem_nan, 75)
    iqr_value = q3 - q1
    
    lower_bound = q1 - 1.5 * iqr_value
    upper_bound = q3 + 1.5 * iqr_value

Desta forma, valores acima do limite superior e abaixo do limite inferior, não entram para o cálculo da variancia.

##### 3.1.4) Cálculo da variância

Definido quais valores serão usados, sem a presença de outliers, foi calculada a variância de cada país. Foi implementado uma regra para caso a **variancia** fosse ***maior do que o valor X (determinado individualmente conforme o comportamento do país)***, seria feito o **cálculo da mediana**. Desta forma, realizando um *segundo filtro de possíveis outliers*. 

##### 3.1.5) Aplicando a variância do país na média móvel do grupo

Para finalmente preencher os valores nulos, foi somada (ou subtraida) a variância (ou mediana), da média móvel do grupo. Desta forma criando uma curva identica a média móvel, porém, com deslocamento da curva. 

Para definir se o valor encontrado da *variância/mediana* seria somado ou subtraído, foi feita uma simples verificação:

- Caso os valores não nulos do país para cada ano fossem maiores do que a média móvel, a *variância/mediana* será somada a própria média móvel.
- Caso contrário, a *variância/mediana* será subtraida.

#### 3.2) Preenchendo os valores do futuro

Para preencher os valores faltantes do futuro, que ainda não representam a predição, mas sim um método para lidar com os dados faltantes (conforme exigido no desafio), foram aplicadas as mesmas regras descritas no tópico *3.1) Preenchendo os valores do passado*. 

A única exceção para este caso foi que o cálculo da média móvel foi realizado com base em **todos** os valores do dataset, **após o tratamento dos dados nulos do passado**. 

### 4) Definição do método de predição

#### 4.1) Seasonal Decompose
Foi feita utilziado o seasonal decompose para verificar se as séries apresentavam tendência e sazionalidade. Para isso, foi criada uma lista randomica de 10 paises para verificar estes parametros. 

Dentro do seasonal decompose foi optado por utilizar: 

    period = 10

Esta escolha se deu por conta da análise gráfica e uma verificação de que durante um período de 10 anos ocorrem similaridades entre todos os paises. 

#### 4.2) Criação de métricas

Foi criada uma função para calculo das métricas: mae, rmse e mape.

#### 4.3) Augmented Dickey-Fuller

Foi utilizado o teste ADF para encontrar os países que apresentam sua series de dados como valores estacionários e não estacionários. Todos os modelos foram aplicados de forma separada conforme o tipo de serie de dados.

#### 4.4) Modelo Auto ARIMA

Foi aplicado o modelo do Auto ARIMA com os seguintes parâmetros: 
   
    start_p=0, max_p=2, 
    start_q=0, max_q=2,
    m=10, 
    seasonal=seasonal,
    stationary=stationary,
    d=d, 
    trend= trend,
    test = 'adf',
    error_acCtion='ignore', 
    stepwise=True

Tendo como variáveis: seasonal, stationary, d e trend. Estes parêmtros variam conforme o modelo é aplicado para um tipo de series, sendo ela, *estacionária* ou *não estacionária*. 

#### 4.5) Modelo Simple Exponential Smoothing

Para definir qual seria o valor do parâmetro *'smoothing_level'* do modelo Simple Exponential Smoothing, foi criada uma função que encontra o melhor valor para cada um dos países, com base no seguinte código:

``` python
for i in columns:
    best_mse = float('inf')
        best_smoothing_level = None
        smoothing_levels = np.linspace(0.01, 1, 10)

        for smoothing_level in smoothing_levels:
            model = SimpleExpSmoothing(df_train[i]).fit(smoothing_level=smoothing_level)
            forecast = model.forecast(steps=len(df_test[i]))
            mse = mean_squared_error(df_test[i], forecast)
            
            if mse < best_mse:
                best_mse = mse
                best_smoothing_level = smoothing_level
``` 

## 5) Escolha do modelo final

O modelo escolhido para a predição final foi o ***Simple Exponential Smoothing***. Fatores que leveram a escolha dele:

a) Foi feita uma comparação entre os parâmetros rsme e mape de ambos os modelos e visto o % de vezes que um modelo obteve resultados maiores que o outro. Como resposta, foi obtido:

    Para rsme:

    ARIMA maior que SES: 65.35%
    SES maior que ARIMA: 6.14%
    ARIMA igual ao SES: 28.51%

    ------------------------------------------------------------
    
    Para mape:

    ARIMA maior que SES: 66.67%
    SES maior que ARIMA: 4.82%
    ARIMA igual ao SES: 28.51%

Como os valores de rsme e mape devem ser os menores possíveis, foi visto que o modelo SES agrada mais este fator. 

b) Faciliade de aplicação do modelo para o dataset. O modelo SES rodou com tempo menor que 1 minutos, enquanto o modelo ARIMA demorou cerca de 15 minutos.

## 6) Executando o projeto

### 6.1) Clonar o repositório

` git clone https://github.com/ThainaMoraes/desafio_ds_lighthouse.git` 

### 6.2) Criação do ambiente virtual

`python -m venv venv` 

### 6.3) Ativar a venv no linux

`source venv/bin/activate`

### 6.4) Instalar as dependências

`pip install -r requirements.txt`

### 6.5) Executar os notebooks

#### a) Primeiro notebook

É necessário executar o primeiro notebook chamado: *desafio_ds_fill_na_values.ipynb*. Este irá gerar dois arquivos de saída:

- *df_final_with_no_na_values.csv*  - Este arquivo sera usado dentro do segundo notebook.*
- *filled_values.csv* - O arquivo está localizado na pasta *files* e este é o arquivo final com os dados 'no data' preenchidos, ainda sem o modelo de predição. 

***Foi optado em criar dois arquivos de saída para facilitar a manipulação do dataset dentro do segundo notebook, considerando que foram feitas substituições nos nomes de alguns países e colunas. Porém o arquivo final 'filled_values.csv*' segue conforme o modelo original dos dados, apenas com inputamento dos dados faltantes.** 
#### b) Segundo notebook

O segundo notebook se chama: *desafio_ds_predict_values.ipynb*.

Este notebook gera apenas um arquivo de saída 'predicted.csv' com o modelo final de predição. 
