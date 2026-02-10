from db.models import TodayEconomicNews
from db.scheduler import populate_database
from db.database import create_tables, get_db
import pandas as pd


db = next(get_db())


def main():
    # create_tables()
    # populate_database()
    query = db.query(TodayEconomicNews)
    df = pd.read_sql_query(query.statement, db.bind, params=query.statement.compile().params)
    print(df.head())
    current_date = df.date.dt.date.iloc[0]
    high_impact_events_count = df[df['importance'] == 1].shape[0]
    high_impact_events = df[df['importance'] == 1]['title'].tolist()
        
    if df.shape[0] > 0:
        message = \
        f'Daily market summary for {current_date}: \n' \
        f'High impact events count: {high_impact_events_count} \n' \
        f'High impact events: {'\n -'.join([' '] + high_impact_events)}' 
        print(message)

if __name__ == '__main__':
    main()

