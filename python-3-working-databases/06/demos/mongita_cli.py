import datetime as dt
import requests
import click
from pymongo import MongoClient


MONGO_URI = "mongodb://dev:dev@localhost:27017/?authSource=admin"

def get_db():
    client = MongoClient(MONGO_URI)
    return client.portfolio

def get_coin_price(coin_id: str, currency: str) -> float:
    url = "https://api.coingecko.com/api/v3/simple/price"
    try:
        resp = requests.get(url, params={"ids": coin_id, "vs_currencies": currency}, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        return float(data[coin_id][currency])
    except (requests.RequestException, KeyError, ValueError) as e:
        raise click.ClickException(f"Failed to fetch price for '{coin_id}' in '{currency}': {e}")

@click.group()
def cli():
    """Crypto portfolio CLI"""
    pass

@cli.command()
@click.option("--coin-id", default="bitcoin", show_default=True)
@click.option("--currency", default="usd", show_default=True)
def show_coin_price(coin_id, currency):
    price = get_coin_price(coin_id, currency)
    click.echo(f"The price of {coin_id} is {price:.2f} {currency.upper()}")

@cli.command()
@click.option("--coin-id", required=True)
@click.option("--currency", required=True)
@click.option("--amount", required=True, type=float)
@click.option("--sell/--buy", default=False, show_default=True, help="Mark as a sell; default is buy")
def add_investment(coin_id, currency, amount, sell):
    db = get_db()
    investments = db.investments
    doc = {
        "coin_id": coin_id,
        "currency": currency,
        "amount": amount,
        "sell": bool(sell),
        "timestamp": dt.datetime.now(dt.timezone.utc),
    }
    investments.insert_one(doc)
    click.echo(f"Added {'sell' if sell else 'buy'} of {amount} {coin_id}")

@cli.command()
@click.option("--coin-id", required=True)
@click.option("--currency", required=True)
def get_investment_value(coin_id, currency):
    db = get_db()
    investments = db.investments
    price = get_coin_price(coin_id, currency)
    buys  = sum(d["amount"] for d in investments.find({"coin_id": coin_id, "currency": currency, "sell": False}))
    sells = sum(d["amount"] for d in investments.find({"coin_id": coin_id, "currency": currency, "sell": True}))
    net = buys - sells
    click.echo(f"You own a total of {net} {coin_id} worth {net * price:.2f} {currency.upper()}")

@cli.command()
@click.option("--csv-file", type=click.Path(exists=True, dir_okay=False))
def import_investments(csv_file):
    click.echo("TODO: implement CSV import")

if __name__ == "__main__":
    cli()
