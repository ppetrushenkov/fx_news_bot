from db.models import TodayEconomicNews
from db.scheduler import populate_database
from db.database import create_tables, get_db
import pandas as pd


db = next(get_db())


def main():
    create_tables()
    populate_database()

    query = db.query(TodayEconomicNews)
    df = pd.read_sql_query(query.statement, db.bind, params=query.statement.compile().params)
        
    if not df.empty:
        current_date = df.date.dt.date.iloc[0]
        high_impact_events_count = df[df['importance'] == 1].shape[0]
        high_impact_events = df[df['importance'] == 1]['title'].tolist()

        daily_summary = \
        f'Daily market summary for {current_date}: \n' \
        f'High impact events count: {high_impact_events_count} \n' \
        f'High impact events: {'\n -'.join([' '] + high_impact_events)}' 
        print(daily_summary)

if __name__ == '__main__':
    main()

