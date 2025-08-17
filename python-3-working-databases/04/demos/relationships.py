from typing import List
from sqlalchemy import String, Numeric, create_engine, select, Text, ForeignKey
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, Session, relationship



class Base(DeclarativeBase):
    pass

class Investment(Base):
    __tablename__ = "investment"

    id: Mapped[int] = mapped_column(primary_key=True)
    coin: Mapped[str] = mapped_column(String(32))
    currency: Mapped[str] = mapped_column(String(3))
    amount: Mapped[float] = mapped_column(Numeric(10, 2))

    portfolio_id: Mapped[int] = mapped_column(ForeignKey("portfolio.id"))
    portfolio: Mapped["Portfolio"] = relationship(back_populates="investments")


    def __repr__(self) -> str:
        return f"<Investment coin: {self.coin!r}, currency: {self.currency!r}, amount: {self.amount}>"

class Portfolio(Base):
    __tablename__= "portfolio"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(256))
    description: Mapped[str] = mapped_column(Text())

    investments: Mapped[List[Investment]] = relationship(
        back_populates="portfolio"
    )

    def __repr__(self) -> str:
        return f"<Portfolio name: {self.name} with {len(self.investments)} investments>"

engine = create_engine("sqlite:///demo_r.db", future=True, connect_args={"timeout": 15})
Base.metadata.create_all(engine)

bitcoin = Investment(coin="bitcoin", currency="USD", amount=1.0)
ethereum = Investment(coin="ethereum", currency="GBP", amount=10.0)
dogecoin = Investment(coin="dogecoin", currency="EUR", amount=100.0)

portfolio_1 = Portfolio(name="Portfolio 1", description="Description 1")
portfolio_2 = Portfolio(name="Portfolio 2", description="Description 2")

bitcoin.portfolio = portfolio_1

portfolio_2.investments.extend([ethereum, dogecoin])

portfolio_3 = Portfolio(name="Portfolio 3", description="Description 3")
bitcoin_2 = Investment(coin="bitcoin", currency="USD", amount=2.0)

bitcoin_2.portfolio = portfolio_3

with Session(engine) as session:
    # session.add(bitcoin)
    # session.add(portfolio_2)
    # session.commit()

    portfolio = session.get(Portfolio, 2)

    for investment in portfolio.investments:
        print(investment)

    print(portfolio)

    investment = session.get(Investment, 1)
    print(investment.portfolio)

    stmt = select(Investment).join(Portfolio)
    print(stmt)

    # session.add(bitcoin_2)
    # session.commit()

    subq = select(Investment).where(Investment.coin == "bitcoin").subquery()
    stmt = select(Portfolio).join(subq, Portfolio.id == subq.c.portfolio_id)
    print(stmt)

    portfolios = session.execute(stmt).scalars().all()
    print(portfolios)