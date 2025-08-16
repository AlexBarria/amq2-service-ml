"""
GraphQL API for fashion product retrieval and filtering.

This module defines a FastAPI application with a Strawberry GraphQL schema.
It provides queries for retrieving fashion files and filtering them by attributes,
as well as a mutation for searching similar products using a text description via gRPC.
"""
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
    """
    Calls the gRPC model service to retrieve similar products for a given description.

    Args:
        query (str): Product description.

    Returns:
        ml_service_pb2.Prediction: gRPC response with similar products.
    """
    with grpc.insecure_channel("model_grpc:50051") as channel:
        stub = ml_service_pb2_grpc.MLServiceStub(channel)
        request = ml_service_pb2.Search(description=query, image_path=None)
        response = stub.Predict(request)
    return response


# Strawberry types
@strawberry.type
class FashionFile:
    """
    Strawberry GraphQL type representing a fashion product file.
    """
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
    """
    GraphQL queries for retrieving and filtering fashion files.
    """

    @strawberry.field
    def all_files(self) -> List[FashionFile]:
        """
        Retrieves all fashion files from the database.

        Returns:
            List[FashionFile]: List of all fashion files.
        """
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
        """
        Retrieves fashion files filtered by the provided attributes.

        Args:
            masterCategory (str | None): Filter by master category.
            gender (str | None): Filter by gender.
            baseColour (str | None): Filter by base colour.
            season (str | None): Filter by season.
            year (str | None): Filter by year.
            limit (int): Maximum number of results.
            offset (int): Offset for pagination.

        Returns:
            List[FashionFile]: List of filtered fashion files.
        """
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
    """
    GraphQL mutation for searching similar fashion products by description.
    """

    @strawberry.mutation
    def search(self, description: str) -> list[FashionFile]:
        """
        Searches for similar fashion products using a text description.

        Args:
            description (str): Product description.

        Returns:
            list[FashionFile]: List of similar fashion files.
        """
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
                created_at=prod.created_at
            ) for prod in response.fashion_product]


schema = strawberry.Schema(query=Query, mutation=Mutation)

app = FastAPI()
app.include_router(GraphQLRouter(schema, graphiql=True), prefix="/graphql")
