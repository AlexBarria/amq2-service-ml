import os
from typing import List
from datetime import datetime

import strawberry
from fastapi import FastAPI
from strawberry.fastapi import GraphQLRouter
from sqlalchemy import create_engine, text
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

# SQLAlchemy setup
engine = create_engine(os.getenv("PG_CONN_STR"))

# Strawberry types
@strawberry.type
class FashionFile:
    id: int
    filename: str
    s3_path: str
    masterCategory: str | None
    subCategory: str | None
    articleType: str | None
    baseColour: str | None
    season: str | None
    year: str | None
    usage: str | None
    gender: str | None
    productDisplayName: str | None
    dataset: str | None
    created_at: datetime

@strawberry.type
class Query:
    @strawberry.field
    def all_files(self) -> List[FashionFile]:
        with engine.connect() as conn:
            result = conn.execute(text("SELECT * FROM fashion_files")).fetchall()
            return [FashionFile(**row._mapping) for row in result]

    @strawberry.field
    def files_by_filters(
        self,
        masterCategory: str | None = None,
        gender: str | None = None,
        baseColour: str | None = None,
        season: str | None = None,
        year: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> List[FashionFile]:
        query = "SELECT * FROM fashion_files WHERE 1=1"
        params = {}

        if masterCategory:
            query += ' AND "masterCategory" = :masterCategory'
            params["masterCategory"] = masterCategory
        if gender:
            query += ' AND gender = :gender'
            params["gender"] = gender
        if baseColour:
            query += ' AND "baseColour" = :baseColour'
            params["baseColour"] = baseColour
        if season:
            query += ' AND season = :season'
            params["season"] = season
        if year:
            query += ' AND year = :year'
            params["year"] = year

        query += " ORDER BY id LIMIT :limit OFFSET :offset"
        params["limit"] = limit
        params["offset"] = offset

        with engine.connect() as conn:
            result = conn.execute(text(query), params).fetchall()
            return [FashionFile(**row._mapping) for row in result]

schema = strawberry.Schema(Query)

app = FastAPI()
app.include_router(GraphQLRouter(schema, graphiql=True), prefix="/graphql")
# graphql_app = GraphQLRouter(schema, graphiql=True, debug=True)
# app = FastAPI()
# app.include_router(graphql_app, prefix="/graphql")

# @app.get("/")
# def read_root():
#     return {"message": "FastAPI + Strawberry is running"}