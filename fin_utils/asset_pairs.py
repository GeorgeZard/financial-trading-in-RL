def binance_assets():
    import time
    import requests
    symbols = requests.get("https://api.binance.com/api/v3/exchangeInfo").json()['symbols']
    symbols = {s['symbol']: s for s in symbols}
    from fin_utils.candles.loading import load_feather_dir
    candle_dict = load_feather_dir('~/fin_data/binance/candles/', resample='1D', n_workers=4)

    def get_price_in(base, quote, symbols, candle_dict, intermediates=['BNB', 'BTC', 'ETH', 'USDT']):
        intermediates = intermediates.copy()
        if base in intermediates:
            intermediates.remove(base)
        if quote in intermediates:
            intermediates.remove(quote)
        for v in symbols.values():
            if v['baseAsset'] == base and v['quoteAsset'] == quote:
                return candle_dict[v['symbol'].lower()].close.iloc[-10:].mean()
            elif v['baseAsset'] == quote and v['quoteAsset'] == base:
                return 1. / candle_dict[v['symbol'].lower()].close.iloc[-10:].mean()
        if quote not in intermediates:
            for intermediate in intermediates:
                p1 = get_price_in(base, intermediate, symbols, candle_dict, intermediates)
                p2 = get_price_in(intermediate, quote, symbols, candle_dict, intermediates)
                if p1 > 0 and p2 > 0:
                    return p1 * p2
            return 0

    assetvols = dict()
    for key in candle_dict.keys():
        assetvols[key] = get_price_in(symbols[key.upper()]['baseAsset'], 'USDT', symbols, candle_dict)
    for key in list(assetvols.keys()):
        assetvols[key] = candle_dict[key].volume.iloc[-2000:].sum() * assetvols[key]
    sorted_assets = [k for k, v in sorted(assetvols.items(), key=lambda item: item[1])][::-1]
    return sorted_assets


sorted_assets_binance = [
    'ethusdt', 'bnbusdt', 'ethbtc', 'bnbbtc', 'adausdt', 'btcusdt', 'xrpusdt', 'ethbusd', 'maticbtc', 'adabtc',
    'dogeusdt', 'solusdt', 'busdusdt', 'linkbtc', 'linkusdt', 'dotusdt', 'vetusdt', 'etcusdt', 'solbtc', 'xrpbtc',
    'ltcusdt', 'trxusdt', 'ftmusdt', 'bnbbusd', 'maticusdt', 'lunausdt', 'thetabtc', 'axsusdt', 'eosusdt', 'ftmbtc',
    'bnbeth', 'bchusdt', 'btcbusd', 'trxbtc', 'bttusdt', 'uniusdt', 'chzusdt', 'thetausdt', 'ethusdc', 'etheur',
    'atomusdt', 'wavesbtc', 'xlmusdt', 'iostusdt', 'enjbtc', 'filusdt', 'ltcbtc', 'adabusd', 'neousdt', 'usdcusdt',
    'adaeth', 'oneusdt', 'bqxbtc', 'etcbtc', 'avaxusdt', 'sushiusdt', 'solbusd', 'sxpusdt', 'linketh', 'lunabtc',
    'algousdt', 'dogebtc', 'fetusdt', 'wavesusdt', 'egldusdt', 'dotbtc', 'qtumusdt', 'ethpax', 'vetbtc', 'xtzusdt',
    'hotusdt', 'fetbtc', 'enjusdt', 'aaveusdt', 'rcnbtc', 'xrpbusd', 'fttusdt', 'srmusdt', 'maticbnb', 'xlmbtc',
    'neobtc', 'eosbtc', 'yfiusdt', 'eurusdt', 'tfuelusdt', 'sandusdt', 'omgusdt', 'nearusdt', 'dogebusd', 'manabtc',
    'iotausdt', 'trxeth', 'cakeusdt', 'zilusdt', 'grtusdt', 'atombtc', 'btcusdc', 'ankrusdt', 'hbarusdt', 'runeusdt',
    'zecusdt', 'xmrbtc', 'chzbtc', 'ksmusdt', 'ontusdt', 'dashusdt', 'xmrusdt', 'aliceusdt', 'kavausdt', 'bchbtc',
    'winusdt', 'icxusdt', 'crvusdt', 'usdttry', 'ethtusd', 'lendbtc', 'adabnb', 'adaeur', 'powrbtc', 'shibusdt',
    'npxsusdt', 'adaupusdt', 'renbtc', 'mtlbtc', 'icpusdt', 'ethgbp', 'c98usdt', 'zilbtc', 'icxbtc', 'iotabtc',
    'algobtc', 'dentusdt', 'tfuelbtc', 'celrusdt', 'chrusdt', 'unibtc', 'dotbusd', 'onebtc', 'rvnbtc', 'bnbeur',
    'hbarbtc', 'manausdt', 'maticbusd', 'qtumbtc', 'yfiiusdt', 'wrxbtc', 'alphausdt', 'cotiusdt', 'iostbtc', 'solbnb',
    'btceur', 'batbtc', 'batusdt', 'nanobtc', 'btcpax', 'xvsusdt', 'tlmusdt', 'axsbusd', 'xrpeth', 'xtzbtc', 'sushibtc',
    'dogetry', 'bandusdt', 'adxbtc', 'lunabusd', 'nasbtc', 'rvnusdt', 'wrxusdt', 'omgbtc', 'fttbtc', 'egldbtc',
    'ctsiusdt', '1inchusdt', 'btcupusdt', 'usdcbusd', 'ethdai', 'cakebusd', 'gbpusdt', 'dltbtc', 'veteth', 'dogeeur',
    'reefusdt', 'compusdt', 'skybtc', 'iotxusdt', 'sxpbtc', 'stmxusdt', 'ethtry', 'audiousdt', 'lunabnb', 'ognusdt',
    'cotibtc', 'cvcusdt', 'drepusdt', 'loombtc', 'tusdusdt', 'roseusdt', 'xrpeur', 'bandbtc', 'thetaeth', 'ethupusdt',
    'zrxbtc', 'avaxbtc', 'sandbtc', 'qlcbtc', 'snxusdt', 'renusdt', 'enjeth', 'gobtc', 'bchabcusdt', 'yfibtc',
    'nknusdt', 'bnbusdc', 'bnbupusdt', 'zenusdt', 'xrpupusdt', 'polybtc', 'idexbusd', 'dashbtc', 'elfbtc', 'runebtc',
    'chztry', 'nanousdt', 'zecbtc', 'twtusdt', 'hoteth', 'zrxusdt', 'aavebtc', 'rlcusdt', 'ctsibtc', 'wbtcbtc',
    'appcbtc', 'phxbtc', 'ltcbusd', 'cvcbtc', 'vetbusd', 'mtlusdt', 'tomousdt', 'rlcbtc', 'kncbtc', 'paxusdt', 'sysbtc',
    'srmbtc', 'bakeusdt', 'eurbusd', 'kavabtc', 'ftmbnb', 'axsbtc', 'dockbtc', 'nknbtc', 'cakebnb', 'scusdt', 'npxseth',
    'linkbusd', 'mkrusdt', 'celrbtc', 'rsrusdt', 'filbtc', 'atausdt', 'bcptbtc', 'hntusdt', 'xrpbnb', 'poabtc',
    'storjusdt', 'pivxbtc', 'ambbtc', 'trigbtc', 'yfidownusdt', 'injusdt', 'aebtc', 'xemusdt', 'xembtc', 'usdtdai',
    'steembtc', 'etcbusd', 'bccbtc', 'ontbtc', 'avaxtry', 'btctusd', 'hsrbtc', 'lrcusdt', 'btgbtc', 'sklusdt',
    'bakebnb', 'egldbusd', 'tkousdt', 'eoseth', 'agibtc', 'stxusdt', 'viabtc', 'lrcbtc', 'bakebusd', 'storjbtc',
    'eosbusd', 'vthousdt', 'xecbusd', 'grtbtc', 'erdusdt', 'qspbtc', 'naseth', 'arpausdt', 'bccusdt', 'navbtc',
    'psgusdt', 'dntbtc', 'yoyobtc', 'cmtbtc', 'gntbtc', 'tomobtc', 'ongusdt', 'zenbtc', 'astbtc', 'chrbtc', 'c98busd',
    'oceanusdt', 'xrptry', 'audusdt', 'xrpusdc', 'adausdc', 'bchabcbtc', 'arusdt', 'ognbtc', 'trbusdt', 'bnbtry',
    'dockusdt', 'flmusdt', 'mithusdt', 'etceth', 'ksmbtc', 'reqbtc', 'bnbgbp', 'crvbtc', 'xvgbtc', 'waveseth', 'btttry',
    'xlmdownusdt', 'stxbtc', 'engbtc', 'onebnb', 'rdnbtc', 'unibusd', 'nxsbtc', 'hottry', 'vetbnb', 'celousdt',
    'ethaud', 'litusdt', 'databtc', 'dotbnb', 'sfpusdt', 'ltobtc', 'ostbtc', 'aavedownusdt', 'lendeth', 'kncusdt',
    'hardusdt', 'slpusdt', 'bnbtusd', 'ltceth', 'bchbusd', 'avaxbusd', 'thetabnb', 'wanbtc', 'trxbusd', 'idexbtc',
    'snmbtc', 'cakebtc', 'unfiusdt', 'hiveusdt', 'ctkusdt', 'perpusdt', 'bttbnb', 'datausdt', 'btcgbp', 'manaeth',
    'xrpgbp', 'venbtc', 'ongbtc', 'blzbtc', 'dexebusd', 'jstusdt', 'fttbnb', 'rayusdt', 'nearbtc', 'tntbtc', 'linausdt',
    'trxbnb', 'audiobtc', 'neblbtc', 'bntbtc', 'dodousdt', 'iosteth', 'mftusdt', 'neoeth', 'gtousdt', 'xvsbtc',
    'shibbusd', 'gasbtc', 'iotxbtc', 'sxpbusd', 'adatry', 'dogegbp', 'alphabtc', 'runebusd', 'ethbrl', 'busdtry',
    'degousdt', 'blzusdt', 'truusdt', 'bntusdt', 'adagbp', 'epsusdt', 'brdbtc', 'ankrbtc', 'usdtbrl', 'mblusdt',
    'algobusd', 'dgdbtc', 'filbusd', 'onebusd', 'lskbtc', 'dcrbtc', 'dgbusdt', 'cndbtc', 'doteur', 'keyusdt', 'bcdbtc',
    'perlusdt', 'sntbtc', 'snxbtc', 'compbtc', 'linkusdc', 'arkbtc', 'soltry', 'arnbtc', 'gbpbusd', 'btttrx',
    'superusdt', 'axsbnb', 'zileth', 'dotupusdt', 'btcdownusdt', 'yfiibtc', 'sushibusd', 'bzrxusdt', 'viteusdt',
    'tlmbusd', 'bnbpax', 'belusdt', 'xlmeth', 'poebtc', 'lunaeur', 'lskusdt', 'fuelbtc', 'atombusd', 'funusdt',
    'pondusdt', 'alicebusd', 'arpabtc', 'grsbtc', 'rosebtc', 'ckbusdt', 'edobtc', 'antusdt', 'chzbnb', 'maticeur',
    'maskusdt', 'enjbusd', 'ltousdt', 'ltcdownusdt', 'powreth', 'wtcbtc', 'adadownusdt', 'xlmbusd', 'bcnbtc', 'utkusdt',
    'ltcbnb', 'ncasheth', 'diausdt', 'atmusdt', 'akrousdt', 'mboxusdt', 'minausdt', 'bttbusd', 'pntusdt', 'xtzupusdt',
    'straxusdt', 'srmbusd', 'ardrbtc', 'rcneth', 'aionbtc', 'injbtc', 'dexeeth', 'bnbdownusdt', 'betheth', 'qntusdt',
    'qkcbtc', 'rpxbtc', 'runebnb', 'ornusdt', 'fetbnb', 'icpbusd', 'soleur', 'cosusdt', 'balusdt', 'iotaeth',
    'hbarbusd', '1inchbtc', 'tusdbtc', 'ltcusdc', 'enjbnb', 'nearbusd', 'aionusdt', 'hotbnb', 'veteur', 'xtzdownusdt',
    'ksmbusd', 'sxptry', 'mdabtc', 'icxeth', 'forbusd', 'perlbtc', 'oxtusdt', 'ardrusdt', 'xtzbusd', 'rosebusd',
    'frontbusd', 'fttbusd', 'adxeth', 'repbtc', 'rampusdt', 'chzbusd', 'loometh', 'wanusdt', 'pptbtc', 'kp3rbusd',
    'busddai', 'tvkbusd', 'ftmbusd', 'iotxeth', 'fisusdt', 'wtcusdt', 'tctusdt', 'adxbusd', 'sandbusd', 'fiousdt',
    'prombusd', 'denteth', 'trxusdc', 'xvsbusd', 'mdxusdt', 'lendusdt', 'trxxrp', 'ltceur', 'wavesbusd', 'atabusd',
    'irisusdt', 'hntbtc', 'usdtbidr', 'alicebtc', 'dntusdt', 'kmdbtc', 'btsusdt', 'omgeth', 'sunusdt', 'stptusdt',
    'xrptusd', 'ethbearusdt', 'bnbaud', 'oceanbtc', 'aavebusd', 'ethrub', 'phabusd', 'audbusd', 'beamusdt', 'adaaud',
    'hivebtc', 'ncashbtc', 'xmreth', 'wnxmusdt', 'dgbbtc', 'cocosusdt', 'duskusdt', 'qlceth', 'duskbtc', 'btctry',
    'celrbnb', 'bateth', 'mkrbtc', 'egldbnb', 'vibbtc', 'paxgusdt', 'twtbusd', 'gvtbtc', 'xecusdt', 'omusdt', 'neobusd',
    'busdbrl', 'wingusdt', 'avausdt', 'elfeth', 'btcngn', 'pundixusdt', 'xvgeth', 'umausdt', 'trxtry', 'nmrusdt',
    'knceth', 'icpbtc', 'audiobusd', 'gtcusdt', 'ctsibusd', 'btcstusdt', 'qtumeth', 'repusdt', 'bchsvusdt', 'vitebtc',
    'atombnb', 'yfibusd', 'linktusd', 'wavesbnb', 'ghstbusd', 'etcbnb', 'wabibtc', 'troyusdt', 'sklbtc', 'keepusdt',
    'evxbtc', 'xvsbnb', 'sxpupusdt', 'adabrl', 'linkdownusdt', 'iotabusd', 'btcaud', 'unibnb', 'osteth', 'ankrbnb',
    'nanoeth', 'burgerusdt', 'zilbusd', 'saltbtc', 'ctsibnb', 'cdtbtc', 'mithbtc', 'qspeth', 'ethdownusdt', 'nulsbtc',
    'twtbtc', 'vidtbusd', 'trxdownusdt', 'chrbnb', 'ctxcusdt', 'lrceth', 'tlmbtc', 'tnbbtc', 'stmxeth', 'sxpdownusdt',
    'juvusdt', 'nulsusdt', 'lunbtc', 'rvntry', 'linkeur', 'trbbtc', 'bnbbrl', 'gxsusdt', 'zrxeth', 'wbtceth',
    'tfuelbnb', 'usdtrub', 'iqbusd', 'irisbtc', 'sceth', 'diabtc', 'gtobtc', 'dydxusdt', 'xrpdownusdt', 'sysbusd',
    'wingsbtc', 'dexeusdt', 'sushiupusdt', 'dogebnb', 'xrpaud', 'linkupusdt', 'mthbtc', 'oaxbtc', 'manabusd', 'wrxbnb',
    'mdtusdt', 'pondbusd', 'neobnb', 'aaveeth', 'kmdusdt', 'btcbrl', 'vetgbp', 'mtleth', 'subbtc', 'badgerusdt',
    'ltctusd', 'acmusdt', 'qtumbusd', 'mirusdt', 'bchsvbtc', 'avabtc', 'sandbnb', 'cmteth', 'eosbnb', 'sushidownusdt',
    'bchusdc', 'dcrusdt', 'bnbdai', 'zilbnb', 'ogusdt', 'xrppax', 'arbusd', 'winbnb', 'pivxeth', 'iotxbusd',
    'ltcupusdt', 'iostbusd', 'insbtc', 'phbbtc', 'wintrx', 'algobnb', 'clvusdt', 'dottry', 'iostbnb', 'crvbusd',
    'rlceth', 'alphabusd', 'rsrbtc', 'mfteth', 'eosupusdt', 'adatusd', 'dogeaud', 'funbtc', 'mcobtc', 'ethbidr',
    'xlmbnb', 'bnbbidr', 'eosusdc', 'trxtusd', 'injbnb', 'modbtc', 'straxbtc', 'fueleth', 'btsbtc', 'dogebrl', 'c98btc',
    'npxsusdc', 'forthusdt', 'hegicbusd', 'veneth', 'vidtbtc', 'psgbtc', 'dashbusd', 'perpbusd', 'celobtc', 'filbnb',
    'gobnb', 'tusdbusd', 'steemeth', 'snglsbtc', 'mboxbusd', 'slpeth', 'btgusdt', 'reefbusd', 'pntbtc', 'utkbtc',
    'vibebtc', 'aergobtc', 'sushibnb', 'avaxbnb', 'sxpbnb', 'chzeur', 'icxbusd', 'erdbtc', 'neotry', 'ernusdt',
    'uniupusdt', 'flowusdt', 'firousdt', 'iotabnb', 'wavestusd', 'glmbtc', 'tkobusd', 'compbusd', 'xlmeur', 'beambtc',
    'linabusd', 'unfibtc', 'antbtc', 'aergobusd', 'usdtngn', 'bttusdc', 'gxsbtc', 'hbarbnb', 'creambusd', 'nuusdt',
    'raybusd', 'ontbusd', 'nbsusdt', 'aeeth', 'tvkbtc', 'nanobusd', 'slpbusd', 'rvnbusd', 'keybtc', 'rampbusd',
    'xvgbusd', 'asrusdt', 'shibtry', 'dasheth', 'bnteth', 'egldeur', 'stptbtc', 'injbusd', 'balbtc', 'bchbnb', 'grteth',
    'bzrxbtc', 'ethbullusdt', 'fiobtc', 'wprbtc', 'engeth', 'auctionbusd', 'xvgusdt', 'tkobtc', 'rcnbnb', 'hsreth',
    'cndeth', 'cvceth', 'flmbtc', 'barusdt', 'bandbnb', 'unidownusdt', 'busdbidr', 'dodobusd', 'tornusdt', 'perpbtc',
    'bearusdt', 'burgerbnb', 'eostusd', 'sxpeur', 'mkrbusd', 'bcneth', 'autousdt', 'gnteth', 'srmbnb', 'zecbusd',
    'btcdai', 'rvnbnb', 'snxbusd', 'phbtusd', 'reqeth', 'scrtbtc', 'dgdeth', 'dotgbp', 'erdbnb', 'arbtc', 'eostry',
    'nmrbtc', 'xrpbrl', 'wrxbusd', 'fxsbusd', 'trubtc', 'databusd', 'vettry', 'waxpusdt', 'brdeth', 'dlteth',
    'bullusdt', 'ornbtc', 'filupusdt', 'trigeth', 'qkceth', 'zeneth', 'psgbusd', 'hoteur', 'swrvbusd', 'bnbngn',
    'idexusdt', 'maskbusd', 'belbtc', 'litbusd', 'superbusd', 'adapax', 'ctxcbtc', 'polsusdt', 'litbtc', 'xmrbusd',
    'jstbtc', 'nebleth', 'dfbusd', 'shibeur', 'bnbrub', 'cotibnb', 'batbusd', 'stratbtc', 'paxgbtc', 'zeceth', 'ethngn',
    'sfpbtc', 'qntbusd', 'funeth', 'hotbusd', 'sklbusd', 'wnxmbtc', 'usdtuah', 'neotusd', 'snteth', 'fildownusdt',
    'bcheur', 'onteth', 'ognbnb', 'mftbtc', 'cfxusdt', '1inchbusd', 'easyeth', 'minabusd', 'xlmtry', 'spartabnb',
    'tvkusdt', 'c98bnb', 'oceanbusd', 'nearbnb', 'poeeth', 'epsbusd', 'aaveupusdt', 'atabtc', 'lptusdt', 'avaxeur',
    'onttry', 'mdxbusd', 'winusdc', 'tnbeth', 'alphabnb', 'btttusd', 'uftbusd', 'icnbtc', 'chatbtc', 'cloakbtc',
    'dnteth', 'batbnb', 'unfibusd', 'keyeth', 'mlnusdt', 'rifusdt', 'renbnb', 'icxbnb', 'oxtbtc', 'sfpbusd', 'umabtc',
    'alpacausdt', 'yoyoeth', 'appceth', 'mftbnb', 'stxbnb', 'degobusd', 'bchtusd', 'hardbtc', 'poaeth', 'pondbtc',
    'fisbusd', 'cndbnb', 'powrbnb', 'thetabusd', 'mithbnb', 'ctkbtc', 'cdteth', 'atomusdc', 'vthobnb', 'tnteth',
    'rsrbusd', 'chrbusd', 'kavabnb', 'blzeth', 'rdneth', 'mdtbtc', 'paxbusd', 'yfiibusd', 'linkpax', 'algotusd',
    'ksmbnb', 'naveth', 'waneth', 'degobtc', 'xmrbnb', 'xemeth', 'docketh', 'atmbtc', 'wingbtc', 'mblbnb', 'prombnb',
    'grteur', 'ltcpax', 'xlmupusdt', 'iqbnb', 'trxeur', 'drepbnb', 'btcrub', 'qntbtc', 'scbnb', 'aavebnb', 'farmusdt',
    'tkobidr', 'stmxbnb', 'omgbusd', 'trxpax', 'xtzbnb', 'lrcbusd', 'thetaeur', 'dotdownusdt', 'tctbtc', 'phausdt',
    'chzbrl', 'ckbbusd', 'eoseur', 'botbtc', 'xzcbtc', 'botbusd', 'btteur', 'linktry', 'trxupusdt', 'zenbnb', 'avabusd',
    'xrprub', 'bcceth', 'forbtc', 'trubusd', 'grtbusd', 'linkgbp', 'busdngn', 'proseth', 'matictry', 'bcpteth',
    'hardbusd', 'frontbtc', 'chzgbp', 'dataeth', 'galausdt', 'xlmtusd', 'solbrl', 'busdrub', 'antbusd', 'rifbtc',
    'rampbtc', 'ctkbusd', 'superbtc', 'ombusd', 'skyeth', 'scbtc', 'storjeth', 'enjeur', 'oneusdc', 'phxeth',
    'dydxbusd', 'klayusdt', 'phabtc', 'lsketh', 'fiobusd', 'tomobusd', 'cotibusd', 'rpxeth', 'agieth', 'epsbtc',
    'bifibusd', 'cosbtc', 'coverbusd', 'diabusd', 'adxbnb', 'tribeusdt', 'paxbtc', '1inchupusdt', 'zrxbusd', 'celrbusd',
    'tusdeth', 'auctionbtc', 'agixbtc', 'neousdc', 'asteth', 'qspbnb', 'forusdt', 'vibeth', 'arpabnb', 'glmeth',
    'bondusdt', 'busdvai', 'eosbearusdt', 'arneth', 'repeth', 'nanobnb', 'aioneth', 'icpbnb', 'bnbidrt', 'ethzar',
    'juvbtc', 'yfieur', 'ftmusdc', 'ombtc', 'cosbnb', 'usdcpax', 'nmrbusd', 'linabtc', 'bandbusd', 'solgbp', 'dodobtc',
    'firobtc', 'perlbnb', 'dgbbusd', 'bchpax', 'atabnb', 'ogbtc', 'adarub', 'btcstbusd', 'ambeth', 'ezbtc', 'nknbnb',
    'badgerbtc', 'fisbtc', 'cvpbusd', 'yoyobnb', 'tornbusd', 'kncbusd', 'minabtc', 'syseth', 'kp3rbnb', 'mirbusd',
    'etceur', 'xlmusdc', 'skybnb', 'eospax', 'hcbtc', 'ontbnb', 'yfibnb', 'mirbtc', 'bnbusds', 'stmxbtc', 'clvbusd',
    'ltcgbp', 'burgerbusd', 'keepbusd', 'blzbnb', 'galabusd', 'asrbtc', 'scrteth', 'crvbnb', 'bccbnb', 'avabnb',
    'waxpbusd', 'fxsbtc', 'bakebtc', 'kmdeth', 'btcstbtc', 'bzrxbusd', 'ghsteth', 'winbusd', 'tomobnb', 'snmeth',
    'wavesusdc', 'gxseth', 'belbusd', 'rlcbnb', 'flowbusd', 'straxbusd', 'trbbusd', 'bchdownusdt', 'edoeth',
    'pundixeth', 'jstbnb', 'icpeur', 'steembnb', 'dashbnb', 'gtcbusd', 'eosbullusdt', 'dltbnb', 'creambnb', 'usdtidrt',
    'btgeth', 'bchabctusd', 'acmbtc', 'wanbnb', 'badgerbusd', 'zecusdc', 'dotbidr', 'mlnbusd', 'ernbusd', 'fronteth',
    'antbnb', 'acmbusd', 'dotbrl', 'snxbnb', 'xzcusdt', 'ethbearbusd', 'btcbidr', 'viaeth', 'zecbnb', 'easybtc',
    'xlmpax', 'sxpgbp', 'fiobnb', 'nubtc', 'loombnb', 'wineur', 'yfiibnb', 'jstbusd', 'tomousdc', 'alpacabusd',
    'bttpax', 'xrpbearusdt', 'stormbtc', 'linkaud', 'mboxbtc', 'juvbusd', 'bcnbnb', 'quickbusd', 'bchupusdt', 'salteth',
    'arbnb', 'qtumbnb', 'venusdt', 'mdaeth', 'wtceth', 'mdxbtc', 'nubusd', 'dntbusd', 'kavabusd', 'unieur', 'mkrbnb',
    'enjgbp', 'paxtusd', 'btcusds', 'hegiceth', 'rsrbnb', 'polsbusd', 'subeth', 'flmbusd', 'keepbtc', 'linkbrl',
    'bntbusd', 'grttry', 'farmbusd', 'maskbnb', 'ltcbrl', 'ctkbnb', 'trigbnb', 'bcptbnb', 'usdctusd', 'nxseth',
    'straxeth', 'bttbrl', 'gtoeth', 'yfiupusdt', 'wabibnb', 'flowbtc', 'icneth', 'brdbnb', 'swrvbnb', 'phxbnb',
    'oceanbnb', 'troybnb', 'wingseth', 'forthbusd', 'dogerub', 'nasbnb', 'polybnb', 'bcdeth', 'dgbbnb', 'batusdc',
    'arketh', 'mcoeth', 'modeth', 'eosdownusdt', 'appcbnb', 'drepbtc', 'cmtbnb', 'autobusd', 'polsbtc', 'adabidr',
    'utkbusd', 'qlcbnb', 'cfxbusd', 'susdusdt', 'forthbtc', 'ethuah', 'wingbusd', 'bnbzar', 'dockbusd', 'tusdbnb',
    'evxeth', 'lptbtc', 'requsdt', 'vidtusdt', 'ncashbnb', 'stmxbusd', 'ftmtusd', 'venbnb', 'ezeth', 'winbrl', 'barbtc',
    'mboxbnb', 'cocosbnb', 'zectusd', 'balbusd', 'grseth', 'clvbtc', 'sxpaud', 'poabnb', 'agibnb', 'runeeur', 'ltobusd',
    'wtcbnb', 'phbbnb', '1inchdownusdt', 'aebnb', 'hntbusd', 'pivxbnb', 'paxgbnb', 'neblbnb', 'quickusdt', 'ppteth',
    'renbtcbtc', 'maticgbp', 'ufteth', 'mtheth', 'cakegbp', 'btcuah', 'belbnb', 'unfibnb', 'atmbusd', 'stratusdt',
    'xrpngn', 'vgxbtc', 'lptbusd', 'troybtc', 'sysbnb', 'hardbnb', 'barbusd', 'nulseth', 'bifibnb', 'gtcbtc', 'akrobtc',
    'bnbbullusdt', 'zrxbnb', 'autobtc', 'polyusdt', 'btgbusd', 'xrpbullusdt', 'sxpbidr', 'scbusd', 'ostbnb', 'enjbrl',
    'arpatry', 'rdnbnb', 'tribebusd', 'viabnb', 'mdxbnb', 'stormusdt', 'snglseth', 'mcousdt', 'nxsbnb', 'bchabcusdc',
    'gvteth', 'bearbusd', 'gntbnb', 'rpxbnb', 'wnxmbnb', 'ltcrub', 'qntbnb', 'wpreth', 'bullbusd', 'ethbullbusd',
    'bchabcpax', 'chateth', 'onetusd', 'raybnb', 'polybusd', 'gnousdt', 'ambbnb', 'xembnb', 'ardreth', 'waxpbtc',
    'onebidr', 'dydxbtc', 'erdbusd', 'nulsbnb', 'diabnb', 'nmrbnb', 'maticaud', 'eosbearbusd', 'cfxbtc', 'mlnbtc',
    'zenbusd', 'dogebidr', 'stormeth', 'tfueltusd', 'wrxeur', 'cvpeth', 'tornbtc', 'susdbtc', 'zilbidr', 'cloaketh',
    'wabieth', 'bnbuah', 'ankrpax', 'vibeeth', 'bondbtc', 'maticbrl', 'btseth', 'dcrbnb', 'rlcbusd', 'minabnb',
    'inseth', 'navbnb', 'usdtbvnd', 'ghstusdt', 'oaxeth', 'wingbnb', 'xembusd', 'firoeth', 'luneth', 'farmbtc',
    'atomtusd', 'bondbusd', 'onepax', 'klaybtc', 'klaybusd', 'busduah', 'elfusdt', 'ftmpax', 'dfeth', 'polsbnb',
    'trxngn', 'gtobnb', 'covereth', 'neopax', 'strateth', 'vthobusd', 'maticbidr', 'bnbbearusdt', 'aionbnb', 'hcusdt',
    'tfuelusdc', 'dogeusdc', 'cvcbnb', 'algopax', 'bcptusdc', 'solaud', 'omgbnb', 'etcusdc', 'quickbtc', 'xrpbearbusd',
    'btcidrt', 'lskbnb', 'usdsusdt', 'runegbp', 'algousdc', 'dotaud', 'ornbusd', 'flowbnb', 'trxaud', 'axsbrl',
    'xzceth', 'wavespax', 'daiusdt', 'usdpusdt', 'btczar', 'shibbrl', 'usdspax', 'ongbnb', 'keepbnb', 'etcbrl',
    'ernbnb', 'usdsusdc', 'compbnb', 'ankrusdc', 'nubnb', 'alpacabtc', 'btsbnb', 'lendbusd', 'gnobusd', 'dcrbusd',
    'tornbnb', 'etctusd', 'storjbusd', 'duskbnb', 'atompax', 'irisbnb', 'clvbnb', 'battusd', 'lptbnb', 'dydxbnb',
    'usdtzar', 'ardrbnb', 'dotrub', 'mdtbnb', 'usdpbusd', 'klaybnb', 'mlnbnb', 'renbtceth', 'ltobnb', 'hivebnb',
    'irisbusd', 'dogepax', 'paxbnb', 'usdstusd', 'repbnb', 'galabtc', 'ltcngn', 'tfuelpax', 'etcgbp', 'ognbusd',
    'trurub', 'maticrub', 'trbbnb', 'runeaud', 'tribebtc', 'btcbbtc', 'paxeth', 'waxpbnb', 'alpacabnb', 'mcobnb',
    'bchsvtusd', 'stormbnb', 'gtcbnb', 'solrub', 'srmbidr', 'hceth', 'xzcbnb', 'wnxmbusd', 'busdbvnd', 'galabnb',
    'eosbullbusd', 'elfbusd', 'vgxeth', 'aionbusd', 'ontpax', 'quickbnb', 'duskusdc', 'etcpax', 'ankrtusd', 'zecpax',
    'batpax', 'phbusdc', 'flmbnb', 'daibtc', 'vitebnb', 'duskpax', 'cakebrl', 'hotbrl', 'balbnb', 'busdzar',
    'xrpbullbusd', 'erdusdc', 'btsbusd', 'xzcxrp', 'bnbbullbusd', 'farmbnb', 'btcvai', 'gnobtc', 'bzrxbnb', 'axsaud',
    'beambnb', 'ethusdp', 'shibrub', 'usdtgyen', 'prombtc', 'icprub', 'bnbbearbusd', 'bondbnb', 'busdidrt', 'repbusd',
    'phbpax', 'stratbusd', 'tribebnb', 'paxgbusd', 'gtousdc', 'mtlbusd', 'stptbnb', 'daibusd', 'ontusdc', 'linkngn',
    'eosaud', 'erdpax', 'bchsvusdc', 'bcpttusd', 'gnobnb', 'bchabcbusd', 'susdeth', 'blzbusd', 'bchsvpax', 'tctbnb',
    'ctxcbnb', 'usdsbusds', 'aavebrl', 'gtotusd', 'btcusdp', 'usdcbnb', 'bcptpax', 'kmdbusd', 'fisbrl', 'nbsbtc',
    'xrpbidr', 'perlusdc', 'solbidr', 'tlmtry', 'bnbusdp', 'c98brl', 'usdsbusdt', 'stratbnb', 'daibnb', 'sushibidr',
    'gtopax', 'sunbtc', 'dotngn', 'axsbidr', 'btcgyen', 'mblbtc', 'winbtc', 'bttbtc', 'ckbbtc', 'bgbpusdc', 'hotbtc',
    'reefbtc', 'npxsbtc', 'cocosbtc', 'bchabusd', 'dentbtc', 'tusdbtusd', 'bqxeth']
1

if __name__ == '__main__':
    pairs = binance_assets()
    print(pairs)
