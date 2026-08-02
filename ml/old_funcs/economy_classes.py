MONETARY_POLICY = [
    "rate", "policy", "monetary", "fomc", "mpc", "meeting",
    "minutes", "statement", "votes",
    "asset purchase", "purchase facility", "refinancing",
    "cash rate", "bank rate", "official cash",
    "fed", "ecb", "boe", "boj", "rba", "rbnz", "boc", "snb",
    "overnight rate", "federal funds", "funds rate",
    "interest rate", "deposit interest rate", "loan prime rate", 
    "central bank balance sheet"
]

CB_SPEECH = [
    "speaks", "press conference", "press",
    "chair", "president", "gov",
    "powell", "lagarde", "bailey", "kuroda", "draghi",
    "yellen", "carney", "bullard", "waller", "kashkari",
    "member"
]

INFLATION = [
    "cpi", "core cpi", 
    "ppi", "core ppi",
    "pce", "core pce",
    "inflation rate", "core inflation rate",
    "consumer price index",
    "harmonized inflation rate", "pce price index",
    "inflation", "price index", "prices",
    "median cpi", "trimmed cpi",
    "inflation expectations", "tokyo cpi"
]

LABOR_MARKET = [
    "employment", "unemployment", "job", "claims",
    "employment change", "averate hourly earnings",
    "non farm", "nonfarm", "payrolls",
    "adp", "jolts", "job openings", "jobless",
    "labor", "labor force", "labor market",
    "earnings", "hourly earnings",
    "job cuts", "challenger"
]

GDP = [
    "gdp", "gdp growth rate", "gdp annual",
    "growth annualized", "gdp qoq", "monthly gdp"
]

ECONOMIC_ACTIVITY = [
    "manufacturing", "services", 
    "output", "capacity", "utilization",
    "productivity", "investment"
]

SENTIMENT = [
    "pmi", "ism", "confidence", "sentiment",
    "ifo", "zew", "gfk", "nfib", "philly",
    "survey", "optimism", "barometer",
    "expectations", "watchers"
]

CONSUMER_HOUSING = [
    "retail", "sales", "consumer",
    "housing", "home", "mortgage",
    "building", "permits", "starts",
    "house price", "hpi", 
    "personal spending", "personal income",
    "new home sales", "existing home sales", 
    "building permits", "housing starts"
]

MANUFACTORING = [
    "industrial", "production",
    "orders", "factory orders",
    "goods"
]

TRADE_FINANCE = [
    "trade", "balance", "current account",
    "budget", "borrowing", "credit",
    "money supply", "m2", "m3",
    "lending", "loans", "reserves",
    "foreign", "currency"
]

COMMODITIES = [
    "oil", "crude", "gas",
    "inventories", "storage",
    "opec", "commodity prices"
]

BONDS = [
    '10y bond yield',
    'bond', 'bund ', 'yields',
    'yield', 'bonds', 
    'bond yield', '2y note yield'
]


CLASS_KEYWORDS = {
        "MONETARY_POLICY": MONETARY_POLICY,
        "CB_SPEECH": CB_SPEECH,
        "INFLATION": INFLATION,
        "LABOR_MARKET": LABOR_MARKET,
        "GDP": GDP,
        "ECONOMIC_ACTIVITY": ECONOMIC_ACTIVITY,
        "SENTIMENT": SENTIMENT,
        "CONSUMER_HOUSING": CONSUMER_HOUSING,
        "MANUFACTORING": MANUFACTORING,
        "TRADE_FINANCE": TRADE_FINANCE,
        "COMMODITIES": COMMODITIES,
        "BONDS": BONDS
    }