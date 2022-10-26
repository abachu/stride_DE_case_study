import streamlit as st
import psycopg2
import pandas as pd
import pandas.io.sql as sqlio
import altair as alt
from environs import Env
from sqlalchemy import create_engine, text
import streamlit.components.v1 as components
from pandas.api.types import (
    is_categorical_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_object_dtype,
)

def filter_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a UI on top of a dataframe to let viewers filter columns

    Args:
        df (pd.DataFrame): Original dataframe

    Returns:
        pd.DataFrame: Filtered dataframe
    """
    modify = st.checkbox("Add filters")

    if not modify:
        return df

    df = df.copy()

    # Try to convert datetimes into a standard format (datetime, no timezone)
    for col in df.columns:
        if is_object_dtype(df[col]):
            try:
                df[col] = pd.to_datetime(df[col])
            except Exception:
                pass

        if is_datetime64_any_dtype(df[col]):
            df[col] = df[col].dt.tz_localize(None)

    modification_container = st.container()

    with modification_container:
        to_filter_columns = st.multiselect("Filter dataframe on", df.columns)
        for column in to_filter_columns:
            left, right = st.columns((1, 20))
            # Treat columns with < 10 unique values as categorical
            if is_categorical_dtype(df[column]) or df[column].nunique() < 10:
                user_cat_input = right.multiselect(
                    f"Values for {column}",
                    df[column].unique(),
                    default=list(df[column].unique()),
                )
                df = df[df[column].isin(user_cat_input)]
            elif is_numeric_dtype(df[column]):
                _min = float(df[column].min())
                _max = float(df[column].max())
                step = (_max - _min) / 100
                user_num_input = right.slider(
                    f"Values for {column}",
                    min_value=_min,
                    max_value=_max,
                    value=(_min, _max),
                    step=step,
                )
                df = df[df[column].between(*user_num_input)]
            elif is_datetime64_any_dtype(df[column]):
                user_date_input = right.date_input(
                    f"Values for {column}",
                    value=(
                        df[column].min(),
                        df[column].max(),
                    ),
                )
                if len(user_date_input) == 2:
                    user_date_input = tuple(map(pd.to_datetime, user_date_input))
                    start_date, end_date = user_date_input
                    df = df.loc[df[column].between(start_date, end_date)]
            else:
                user_text_input = right.text_input(
                    f"Substring or regex in {column}",
                )
                if user_text_input:
                    df = df[df[column].astype(str).str.contains(user_text_input)]

    return df

env = Env()

# Some Postgres data processing came from:
# @source https://docs.streamlit.io/knowledge-base/tutorials/databases/postgresql


@st.cache(allow_output_mutation=True, hash_funcs={"_thread.RLock": lambda _: None})
def init_connection():
    """
    Initialize connection with the SQLAlchemy engine

    Uses st.cache to only run once.
    """
    db_host = env.str("POSTGRES_HOST")
    db_user = env.str("POSTGRES_USER")
    db_port = env.int("POSTGRES_PORT")
    db_name = env.str("POSTGRES_DB")
    db_pass = env.str("POSTGRES_PASSWORD")
    engine = create_engine(
        f"postgresql://{db_user}:{db_pass}@{db_host}:{db_port}/{db_name}"
    )
    return engine.connect()


def main():

    conn = init_connection()

    states_df = pd.read_sql_table(
        "zip_code_data", conn)
    # TODO: Order states east-to-west
    zip_by_state_df = (
        states_df.groupby("State")["State"]
        .count()
        .reset_index(name="Total Number of Zipcodes")
    )
    zip_by_state_chart = (
        alt.Chart(zip_by_state_df)
        .mark_circle()
        .encode(x="State:N", y="Total Number of Zipcodes:Q")
    )


    institutions_df = pd.read_sql_table(
        "institution_data", conn)
    i_required_cols = ['UNITID', 'INSTNM', 'CITY', 'STABBR', 'ZIP']
    institutions_df = institutions_df[i_required_cols]



    program_df = pd.read_sql_table(
    "program_data", conn)
    p_required_cols = ['UNITID', 'INSTNM', 'CIPCODE', 'CIPDESC', 'CREDDESC', 'DTE_RATIO']
    program_df = program_df[p_required_cols]
    plist_df = program_df[['CIPDESC', 'DTE_RATIO']]
    plist_df = plist_df.groupby('CIPDESC')['DTE_RATIO'].sum()
    plist_df = plist_df.reset_index()
    plist_df = plist_df.sort_values('DTE_RATIO', ascending=False)
    plist_df = plist_df.reset_index(drop=True)
    plist50_df = plist_df[0:50]

    cs_institutions = program_df[program_df['CIPDESC'] == 'Computer Science.']
    top_cs_institutions = cs_institutions[['INSTNM', 'DTE_RATIO']]
    top_cs_institutions = top_cs_institutions.sort_values('DTE_RATIO', ascending=False).reset_index(drop=True)
    top_cs_institutions = top_cs_institutions.drop_duplicates().reset_index(drop=True)
    top_cs_institutions.index = top_cs_institutions.index + 1

    dte_df = pd.merge(program_df, institutions_df, left_on='UNITID', right_on='UNITID', how='left')
    top_cities_df = dte_df[['CITY', 'DTE_RATIO']]
    top_cities = top_cities_df.groupby('CITY')['DTE_RATIO'].sum().reset_index()
    top_cities = top_cities.sort_values('DTE_RATIO', ascending=False).reset_index(drop=True)
    top_10_cities = top_cities['CITY'][0:10].tolist()
    top_citiesp_df = dte_df[dte_df['CITY'].isin(top_10_cities)]
    top_citiesp_df = top_citiesp_df[['CITY', 'CIPDESC', 'DTE_RATIO']]
    top_10_cities_w_top_programs = top_citiesp_df.groupby('CITY').apply(pd.DataFrame.sort_values, 'DTE_RATIO',
                                                                        ascending=False).reset_index(drop=True)


    # TODO: Render Stride logo
    st.title("Stride Funding - Data Engineering")
    st.write("## Zipcodes by State")
    st.altair_chart(zip_by_state_chart, use_container_width=True)
    st.table(zip_by_state_df)
    st.write("## Top Program List")
    st.table(plist50_df)
    st.write("## Top CS Institutions")
    st.table(top_cs_institutions)
    st.write("## Most Valuable Metro Areas")
    st.dataframe(filter_dataframe(top_10_cities_w_top_programs))
    st.write("## Filtered dataframe")
    st.dataframe(filter_dataframe(dte_df))
    st.write("© Stride Funding, Inc. 2022")


if __name__ == "__main__":
    main()
