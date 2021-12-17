import pandas as pd
from pypfopt import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices
import yfinance as yf
import numpy as np
import datetime
     
#symbols = ["XAR", "XBI", "XES", "XHB", "XHE", "XHS", "XITK", "XLE", "XLB", "XLF", "XLI", "XLK", "XLP", "XLRE", "XLU", "XLV", "XLY", "XME", "XOP", "XPH", "XRT", "XSD", "XSW", "XTH", "XTL", "XTN", "XWEB"]
#symbols = ["HRB", "ODFL",  "DLTR", "AMGN", "VRTX", "ABEV"]
#symbols = ["JCI", "RRD", "CAG", "CSC", "UNH", "FLO", "KO", "MDT"]]
symbols = []
#symbols = ["ARCT", "BNTX", "CSLLY", "DVAX", "GSK", "INO", "JNJ", "MRNA", "NVAX", "PFE", "SNY","TBIO", "ADPT", "ALXN", "ALNY", "AMGN", "BAYRY", "BIIB", "EBS", "GILD", "GRFS", "INCY", "LLY", "NVS", "REGN", "RHHBY", "TAK", "VNDA", "ABT", "BDX", "BMXMF", "CODX", "DHR", "LH", "OPK", "DGX", "TMO"]
symbols00 = ["ONCE"
,"BASI"
,"NSYS"
,"WTBA"
,"GWRS"
,"IBKCP"
,"RMQHX"
,"LDVCX"
,"RBCAA"
,"RMQCX"
,"RMQAX"
,"LDVAX"
,"LITE"
''',"LOAN"'''
,"NBTB"
,"ROLL"
,"LDVIX"
,"HBCP"
,"SELF"
,"SCVJX"
,"YORW"
,"SAFT"
,"GBCI"
,"PAHC"
,"BSCUX"
,"HFWA"
,"MSG"
,"QCRH"
,"WIAEX"
,"SCURX"
,"SFBS"
,"BDSKX"
,"UNAM"
,"HNVRX"
,"DCZRX"
,"DMIDX"
,"PRJIX"
,"BRWUX"
,"JSMD"
,"SSSJX"
,"EGORX"
,"HIBUX"
,"SSSDX"
,"SSSKX"
,"BDSIX"
,"BMGGX"
,"HSLVX"
,"SSMLX"
,"SSMJX"
,"DVZRX"
,"LCNB"
,"GCSUX"
,"GBSYX"
,"XBIFX"
,"BDSAX"
,"PFPCX"
,"PFMDX"
,"BDSCX"
,"BCSSX"
,"HNACX"
,"FNWB"
,"JFRNX"
,"STSNX"
,"UIGIX"
,"WEBK"
,"PFGKX"
,"BLGRX"
,"NWKCX"
,"IMPAX"
,"PRVIX"
,"NWHTX"
,"PFOIX"
,"SSBI"
,"GBSPX"
,"NWKDX"
,"STSOX"
,"CNRVX"
,"NWHOX"
,"NWHQX"
,"WIGRX"
,"TMRLX"
,"BSMIX"
,"NWHPX"
,"WOOQX"
,"YOVIX"
,"CNRUX"
,"NWHZX"
,"LSSNX"
,"PFKAX"
,"EBSB"
,"NWSIX"
,"GBSIX"
,"NWHUX"
,"NWSAX"
,"PFJIX"
,"SCRYX"
,"WOOSX"
,"LUSAX"
,"GBSCX"
,"PBLCX"
,"PBLAX"
,"HGIIX"
,"PFGRX"
,"LSITX"
,"NWKBX"
,"SCRLX"
,"PFACX"
,"SSRRX"
,"PFQDX"
,"WOOOX"
,"SSMHX"
,"PBCKX"
,"SSMKX"
,"BSLNX"
,"GBSAX"
,"IMPLX"
,"SEVPX"
,"ISOZX"
,"LAVVX"
,"JUSOX"
,"SSEYX"
,"QSMNX"
,"LAVSX"
,"SSSYX"
,"CCHRX"
,"QSMLX"
,"AFDZX"
,"FISUX"
,"IMOZX"
,"SPWYX"
,"PFHCX"
,"BSVGX"
,"HNSOX"
,"JGMNX"
,"LAVTX"
,"RTDSX"
,"RTDYX"
,"SSFRX"
,"SSUPX"
,"QACFX"
,"SSSWX"
,"FVC"
,"IEDAX"
,"ISMZX"
,"FISTX"
,"SSSVX"
,"QSERX"
,"JVMTX"
,"JSVRX"
,"QSSRX"
,"MVSSX"
,"DPSYX"
,"BFSAX"
,"RTDTX"
,"WVAIX"
,"IEDZX"
,"IEDRX"
,"BIAWX"
,"FSSMX"
,"PEQSX"
,"DHANX"
,"GENIX"
,"RWEBX"
,"BAFWX"
,"GWEIX"
,"GXXIX"
,"BRSHX"
,"STRNX"
,"GINDX"
,"FUNYX"
,"EQNTX"
,"JSERX"
,"STRLX"
,"BRSBX"
,"VEVRX"
,"DFMIX"
,"ILVOX"
,"BRTNX"
,"DRDAX"
,"VEVYX"
,"EQNUX"
,"BUFDX"
,"BRSJX"
,"BRSSX"
,"EIVFX"
,"EQNAX"
,"RGGYX"
,"EQNIX"
,"DRDIX"
,"RGGKX"
,"BRSPX"
,"BRSTX"
,"EQNSX"
,"BRSYX"
,"PXSAX"
,"JCCIX"
,"JHUCX"
,"WASMX"
,"MRVSX"
,"IWEDX"
,"PXGAX"
,"JAVAX"
,"JHUPX"
,"BIADX"
,"DRDCX"
,"WTDWX"
,"PQSAX"
,"DMVYX"
,"JHUIX"
,"CAGGX"
,"IRSEX"
,"IRIEX"
,"SCSRX"
,"LSCNX"
,"EIVTX"
,"BRSUX"
,"BRSDX"
,"GLFOX"
,"QCIBX"
,"JHQAX"
,"HSEIX"
,"JRVQX"
,"JHUAX"
,"JHQRX"
,"JHQPX"
,"JLKNX"
,"JRINX"
,"PRIKX"
,"JHEQX"
,"JHQCX"
,"LEBOX"
,"HWBK"]
symbols01 = ["RARX","KL", "PLAY","FKRCX","WLWEAX","SAEMX","QRHIX","PREMX","EMTCX", "EMTAX","SAAAX","EMTIX","HCGIX","PEMPX","REBAX", "REBCX"]
symbols10 =["DRNA","GNTY","SSBI","IBKCP","SAMG","HIFS","RNDB","FIBK","HBCP","ENFC","IBD","PVBC","HONE","DURPX","LSXMK","UAE","PBIP","BOKFL","PBBI","SACVX","PBAKX","QUERX","AUENX","PFCDX","GTENX","IICLX","LDRI","PFCCX","IWCLX","RILYL","IACLX","RANGX","IRHIX","RANAX","RANHX","HWBK"]
symbols11 = ["TTD", "HFOTX", "LE", "GNXAX", "UNIT", "HLNE", "SSEQX", "SSELX", "CZR", "HGXCX", "HQY", "PEPFX", "PEIFX", "PJFQX", "BRKRX", "BRKIX", "JARSX", "PEFFX", "BRKSX", "BRKUX", "BRKTX", "ETSY", "QEMAX", "BRKCX", "JATNX"]

portfolio = pd.DataFrame(columns=symbols)
portfolio = portfolio.dropna()
#print(portfolio)
month = 4
day = 1

for symbol in symbols00:
   
    try:
        X = yf.download(symbol,'2020-05-10','2020-05-19')
        X.pop('Open')
        X.pop('High')
        X.pop('Low')
        X.pop('Close')
        X.pop('Volume')
        X['p0'] = np.where(X['Adj Close'] > X['Adj Close'].shift(1), 1, np.where(X['Adj Close'] < X['Adj Close'].shift(1),"0", np.where(X['Adj Close'] == X['Adj Close'].shift(1),"2", "3")))
        X['pattern'] = X['p0'].shift(1).astype(str) + X['p0'].astype(str)# + X['p2'].astype(str) #+ X['p2'].astype(str)

        #X['label'] = np.where(X['pattern'] == '11', 1,-1) 
        #print(X)
        #print(symbol)
        if X['pattern'].tail(1).values == '00':
            if X.index[-1].strftime('%Y-%m-%d') == '2020-05-18':
            #if datetime.datetime.strptime(str(X.index[-1]), '%Y-%m-%d %H:%M:%S').date() == '2020-05-15':
                symbols.append(symbol)
    except Exception as e:
        print(str(e))
        continue

for symbol in symbols01:
   
    try:
        X = yf.download(symbol,'2020-05-10','2020-05-19')
        X.pop('Open')
        X.pop('High')
        X.pop('Low')
        X.pop('Close')
        X.pop('Volume')
        X['p0'] = np.where(X['Adj Close'] > X['Adj Close'].shift(1), 1, np.where(X['Adj Close'] < X['Adj Close'].shift(1),"0", np.where(X['Adj Close'] == X['Adj Close'].shift(1),"2", "3")))
        X['pattern'] = X['p0'].shift(1).astype(str) + X['p0'].astype(str)# + X['p2'].astype(str) #+ X['p2'].astype(str)

        #X['label'] = np.where(X['pattern'] == '11', 1,-1) 
        #print(X)
        #print(symbol)
        if X['pattern'].tail(1).values == '01':
            if X.index[-1].strftime('%Y-%m-%d') == '2020-05-18':
            #if datetime.datetime.strptime(str(X.index[-1]), '%Y-%m-%d %H:%M:%S').date() == '2020-05-15':
                symbols.append(symbol)
    except Exception as e:
        print(str(e))
        continue
    
for symbol in symbols10:
   
    try:
        X = yf.download(symbol,'2020-05-10','2020-05-19')
        X.pop('Open')
        X.pop('High')
        X.pop('Low')
        X.pop('Close')
        X.pop('Volume')
        X['p0'] = np.where(X['Adj Close'] > X['Adj Close'].shift(1), 1, np.where(X['Adj Close'] < X['Adj Close'].shift(1),"0", np.where(X['Adj Close'] == X['Adj Close'].shift(1),"2", "3")))
        X['pattern'] = X['p0'].shift(1).astype(str) + X['p0'].astype(str)# + X['p2'].astype(str) #+ X['p2'].astype(str)

        #X['label'] = np.where(X['pattern'] == '11', 1,-1) 
        #print(X)
        #print(symbol)
        if X['pattern'].tail(1).values == '10':
            if X.index[-1].strftime('%Y-%m-%d') == '2020-05-18':
            #if datetime.datetime.strptime(str(X.index[-1]), '%Y-%m-%d %H:%M:%S').date() == '2020-05-15':
                symbols.append(symbol)
    except Exception as e:
        print(str(e))
        continue
    
for symbol in symbols11:
    try: 
        X = yf.download(symbol,'2020-05-10','2020-05-19')
        X.pop('Open')
        X.pop('High')
        X.pop('Low')
        X.pop('Close')
        X.pop('Volume')
        X['p0'] = np.where(X['Adj Close'] > X['Adj Close'].shift(1), 1, np.where(X['Adj Close'] < X['Adj Close'].shift(1),"0", np.where(X['Adj Close'] == X['Adj Close'].shift(1),"2", "3")))
        X['pattern'] = X['p0'].shift(1).astype(str) + X['p0'].astype(str)# + X['p2'].astype(str) #+ X['p2'].astype(str)

        #X['label'] = np.where(X['pattern'] == '11', 1,-1) 
        #print(X)
        #print(symbol)
        if X['pattern'].tail(1).values == '11':
            if X.index[-1].strftime('%Y-%m-%d') == '2020-05-18':
            #if datetime.datetime.strptime(str(X.index[-1]), '%Y-%m-%d %H:%M:%S').date == '2020-05-15':
                symbols.append(symbol)
    except Exception as e:
        print(str(e))
        continue

for symbol in symbols:

    try:
        X = yf.download(symbol,'1980-11-18','2020-05-19')
        X.pop('Open')
        X.pop('High')
        X.pop('Low')
        X.pop('Close')
        X.pop('Volume')
        portfolio[symbol] = X["Adj Close"]                
    except Exception as e:
        print(str(e))
        continue
       
print("Symbols")
print(symbols)
portfolio.dropna(inplace=True)
#print(portfolio)
# Calculate expected returns and sample covariance
mu = expected_returns.mean_historical_return(portfolio)
#mu = expected_returns.capm_return(portfolio)
#mu = expected_returns.ema_historical_return(portfolio)
#print(mu)
S = risk_models.sample_cov(portfolio)
#S = risk_models.exp_cov(portfolio)

# Optimise for maximal Sharpe ratio
ef = EfficientFrontier(mu, S)
raw_weights = ef.max_sharpe(risk_free_rate=.0061)
#raw_weights = ef.min_volatility()
#raw_weights = ef.max_quadratic_utility()
cleaned_weights = ef.clean_weights()
#ef.save_weights_to_file("weights1.csv")  # saves to file
#print(cleaned_weights)
ef.portfolio_performance(verbose=True,risk_free_rate=.0063)
latest_prices = get_latest_prices(portfolio)

da = DiscreteAllocation(cleaned_weights, latest_prices, total_portfolio_value=33000)
#allocation, leftover = da.lp_portfolio()
allocation, leftover = da.greedy_portfolio()
print("Discrete allocation:", allocation)
print("Funds remaining: ${:.2f}".format(leftover))
