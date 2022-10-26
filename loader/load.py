import datetime
import subprocess
import click
import pandas as pd


class CsvLoader():
    """
    Contains some methods to manipulate or load data.
    """
    def load_zip_code_data(self):
        zip_file = '/data/us-zip-code-latitude-and-longitude.csv'
        click.echo("Starting to load data.")
        cmd = f'csvsql --db postgresql://postgres:postgres@db --insert --chunk-size 100 --tables zip_code_data -v -y 2048 {zip_file}'
        run = subprocess.run(cmd, capture_output=True, shell=True)
        # df = pd.read_csv(zip_file)

        #If we had a non-0 return code.
        if run.returncode:
            raise Exception(f'csvsql exception:', run.stdout, run.stderr)
        else:
            click.echo("Run worked")
            click.echo(run.stdout)
        click.echo("Done loading zip code data...")
        # return df

    def load_institution_data(self):
        zip_file = '/data/CollegeScorecard-Institution-Subset.csv'
        click.echo("Starting to load Institution data")
        cmd = f'csvsql --db postgresql://postgres:postgres@db --insert --chunk-size 100 --tables institution_data -v -y 2048 {zip_file}'
        run = subprocess.run(cmd, capture_output=True, shell=True)

        # If we had a non-0 return code.
        if run.returncode:
            raise Exception(f'csvsql exception:', run.stdout, run.stderr)
        else:
            click.echo("Run worked")
            click.echo(run.stdout)
        click.echo("Done loading institution data...")


    def load_program_data(self):
        zip_file = '/data/CollegeScorecard-Programs-DTE-Subset.csv'
        click.echo("Starting to load Program data")
        cmd = f'csvsql --db postgresql://postgres:postgres@db --insert --chunk-size 100 --tables program_data -v -y 2048 {zip_file}'
        run = subprocess.run(cmd, capture_output=True, shell=True)

        # If we had a non-0 return code.
        if run.returncode:
            raise Exception(f'csvsql exception:', run.stdout, run.stderr)
        else:
            click.echo("Run worked")
            click.echo(run.stdout)
        click.echo("Done loading program data...")



@click.command()
def load():
    """Simple program to load files into our DB."""
    click.echo(f"Loading data sets!")
    # CsvLoader().load_zip_code_data()
    CsvLoader().load_institution_data()
    CsvLoader().load_program_data()
    click.echo("All Done!")

if __name__ == '__main__':
    load()
