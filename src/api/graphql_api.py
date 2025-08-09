import os
from typing import List
from datetime import datetime

import strawberry
from fastapi import FastAPI
from strawberry.fastapi import GraphQLRouter
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import grpc
import ml_service_pb2
import ml_service_pb2_grpc

load_dotenv()

# SQLAlchemy setup
engine = create_engine(os.getenv("PG_CONN_STR"))


def query_model_grpc(query: str):
    with grpc.insecure_channel("model_grpc:50051") as channel:
        stub = ml_service_pb2_grpc.MLServiceStub(channel)
        request = ml_service_pb2.Search(description=query, image=None)
        response = stub.Predict(request)
    return response


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
    embedding: List[float] | None = strawberry.field(default=None)


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


@strawberry.type
class Mutation:
    @strawberry.mutation
    def search(self, description: str) -> list[FashionFile]:
        response = query_model_grpc(description)
        # Assuming response.fashion_product is iterable and matches FashionFile fields
        return [
            FashionFile(
                id=prod.id,
                filename=prod.filename,
                s3_path=prod.s3_path,
                masterCategory=prod.master_category,
                subCategory=prod.sub_category,
                articleType=prod.article_type,
                baseColour=prod.base_colour,
                season=prod.season,
                year=prod.year,
                usage=prod.usage,
                gender=prod.gender,
                productDisplayName=prod.product_display_name,
                dataset=prod.dataset,
                created_at=prod.created_at,
                embedding=list(prod.embedding) if hasattr(prod, "embedding") else None
            ) for prod in response.fashion_product]


schema = strawberry.Schema(query=Query, mutation=Mutation)

app = FastAPI()
app.include_router(GraphQLRouter(schema, graphiql=True), prefix="/graphql")
