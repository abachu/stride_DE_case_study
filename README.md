# Debt to Earnings
This repository holds the source code for Stride's "Debt to Earnings" dashboard.

# How to Run

1. Clone this repository
1. Install Docker and Docker-Compose if you do not yet have them
1. Add raw data files to the directory `data/`
1. Run `docker-compose up --build`

If everything worked you should be able to hit http://localhost:8001 and see your dashboard

# How to Contribute

* Create a private fork of this repository
* Invite Stride to your fork (username [`StrideTechHiring`](https://github.com/StrideTechHiring))
* Create a branch or relevant Pull Requests on your fork showing your work
* Provide instructions for the Stride squad on running your latest work

# Overview

# Architecture

## Containers / Services
 - `db`: a local PostgreSQL database
 - `dashboard`: a simple Streamlit dashboard  
 - `loader`: python basic ETL of raw data to a format used by the dashboard

## Components

This is a fairly simple Data Science dashboard using a few logical components.

- `docker`: used for containerization
- `docker-compose`: used to run multiple Docker containers
- `pandas`: Python Data Science library. Used for basic data analysis
- `postgresql`: database used for SQL querying of the data sets  
- `streamlit`: a Python library for building simple pandas based web pages or dashboards

## Data Sets

This dashboard uses two datasets:

- Price to Earnings
- Zip Codes

## How to Run
`docker-compose up`

Example Run:
```
postgres     | 2021-11-29 21:35:12.786 UTC [1] LOG:  starting PostgreSQL 14.0 (Debian 14.0-1.pgdg110+1) on aarch64-unknown-linux-gnu, compiled by gcc (Debian 10.2.1-6) 10.2.1 20210110, 64-bit
loader       | Loading data set!
loader       | Starting to load data.
postgres     | 2021-11-29 21:35:12.787 UTC [1] LOG:  listening on IPv4 address "0.0.0.0", port 5432
postgres     | 2021-11-29 21:35:12.787 UTC [1] LOG:  listening on IPv6 address "::", port 5432
postgres     | 2021-11-29 21:35:12.788 UTC [1] LOG:  listening on Unix socket "/var/run/postgresql/.s.PGSQL.5432"
postgres     | 2021-11-29 21:35:12.794 UTC [26] LOG:  database system was shut down at 2021-11-29 21:34:46 UTC
postgres     | 2021-11-29 21:35:12.800 UTC [1] LOG:  database system is ready to accept connections
loader       | Done loading data...
loader       | All Done!
loader exited with code 0
dashboard    | 2021-11-29 21:35:15.368 override steps (5) and chunk_size (512) as content does not fit (14 byte(s) given) parameters.
dashboard    | 2021-11-29 21:35:15.368 Trying to detect encoding from a tiny portion of (14) byte(s).
dashboard    | 2021-11-29 21:35:15.371 ascii passed initial chaos probing. Mean measured chaos is 0.000000 %
dashboard    | 2021-11-29 21:35:15.371 ascii should target any language(s) of ['Latin Based']
dashboard    | 2021-11-29 21:35:15.371 ascii is most likely the one. Stopping the process.
dashboard    |
dashboard    |   You can now view your Streamlit app in your browser.
dashboard    |
dashboard    |   Network URL: http://172.24.0.3:80
dashboard    |   External URL: http://144.121.23.18:80
```



# Debugging Steps

*Get a shell to the loader*
`docker-compose run loader bash`

Build a single container
`docker-compose build loader`

*Get a shell to the dashboard*
`docker-compose exec dashboard bash`

*Test a DB connection*
`psql postgres://postgres:postgres@db`

Directly load data
```
csvsql  --db postgresql://db --insert /data/us-zip-code-latitude-and-longitude.cs
```
