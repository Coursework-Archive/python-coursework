from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import csv

import click
import requests
import psycopg2
import psycopg2.extras


@dataclass
class Investment:
    id: int
    coin: str
    currency: str
    amount: float


def get_connection():
    connection = psycopg2.connect(
        host="localhost",
        database="manager",
        user="postgres",
        password="pgpassword"
    )
    return connection


@click.group()
def cli():
    pass


@click.command()
@click.option("--coin", prompt=True)
@click.option("--currency", prompt=True)
@click.option("--amount", prompt=True)
def new_investment(coin, currency, amount):
    stmt = f"""
        insert into investment (
            coin, currency, amount
        ) values (
            '{coin.lower()}', '{currency.lower()}', {amount}
        )
    """
    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute(stmt)
    connection.commit()

    cursor.close()
    connection.close()
    
    print(f"Added investment for {amount} {coin} in {currency}")


@click.command()
@click.option("--filename")
def import_investments(filename):
    stmt = "insert into investment (coin, currency, amount) values %s"

    connection = get_connection()
    cursor = connection.cursor()
    
    with open(filename, 'r') as f:
        coin_reader = csv.reader(f)
        rows = [[x.lower() for x in row[1:]] for row in coin_reader]

    psycopg2.extras.execute_values(cursor, stmt, rows)
    connection.commit()

    cursor.close()
    connection.close()

    print(f"Added {len(rows)} investments")


@click.command()
@click.option("--currency")
def view_investments(currency):
    connection = get_connection()
    cursor = connection.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

    stmt = "select * from investment"

    if currency is not None:
        stmt += " where currency=%s"
        cursor.execute(stmt, (currency.lower(),))
    else:
        cursor.execute(stmt)

    data = [Investment(**dict(row)) for row in cursor.fetchall()]

    cursor.close()
    connection.close()

    if not data:
        print("No investments found.")
        return

    coins = sorted([row.coin.lower().strip() for row in data])
    currencies = set([row.currency.lower().strip() for row in data])

    resp = requests.get(
        "https://api.coingecko.com/api/v3/simple/price",
        params={"ids": ','.join(coins), "vs_currencies": ','.join(currencies)}
    )
    coin_data = resp.json()

    for inv in data:
        coin = inv.coin.lower().strip()
        cur = inv.currency.lower().strip()

        if coin not in coin_data or cur not in coin_data[coin]:
            print(f"SKIPPING {inv.coin}/{inv.currency}: price not available")
            continue

        try:
            amount = Decimal(str(inv.amount))       # from DB decimal/numeric
            price = Decimal(str(coin_data[coin][cur]))  # JSON number -> decimal
        except (InvalidOperation, TypeError, ValueError):
            print(f"Skipping {inv.coin}/{inv.currency}: non-numeric amount/price")
            continue

        total = amount * price
        print(f"{amount} {inv.coin} in {inv.currency} is worth {total}")


cli.add_command(new_investment)
cli.add_command(import_investments)
cli.add_command(view_investments)

if __name__ == "__main__":
    cli()
