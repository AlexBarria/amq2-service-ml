import os
import streamlit as st
import requests
from typing import List, Dict, Any
from dotenv import load_dotenv
import boto3
import io
from PIL import Image

# Load environment variables
load_dotenv()

# Configure page
st.set_page_config(
    page_title="Fashion Product Search",
    page_icon="👕",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants
REST_API_URL = os.getenv("REST_API_URL", "http://localhost:8800")
GRAPHQL_URL = os.getenv("GRAPHQL_URL", "http://localhost:8801/graphql")
S3_ENDPOINT = os.getenv("S3_ENDPOINT", "http://localhost:9000")
S3_ACCESS_KEY = os.getenv("S3_ACCESS_KEY", "")
S3_SECRET_KEY = os.getenv("S3_SECRET_KEY", "")

# Types
class Product:
    def __init__(self, data: Dict[str, Any]):
        self.id = data.get("id")
        self.s3_path = data.get("s3_path") or data.get("s3Path")  
        self.product_display_name = data.get("productDisplayName")
        self.master_category = data.get("masterCategory")
        self.sub_category = data.get("subCategory")
        self.article_type = data.get("articleType")
        self.base_color = data.get("baseColour")
        self.season = data.get("season")
        self.usage = data.get("usage")
        self.gender = data.get("gender")
        self.year = data.get("year")
        
        if self.s3_path:
            encoded_path = self.s3_path.replace('/', '%2F')
            self.image_url = f"http://localhost:9001/browser/prod/{encoded_path}"
        else:
            self.image_url = ""
    

def search_by_description_rest(description: str) -> List[Product]:
    """Search products by description using REST API"""
    try:
        response = requests.post(
            f"{REST_API_URL}/search/description",
            json={"description": description},
            timeout=10
        )
        response.raise_for_status()
        results = response.json()
        if not isinstance(results, list):
            results = [results]  # Ensure we always return a list
        return [Product(item) for item in results]
    except Exception as e:
        st.error(f"Error searching by description: {str(e)}")
        if 'response' in locals():
            st.error(f"Response: {response.text}")
        return []

def search_by_image_rest(image_file) -> List[Product]:
    """Search products by image using REST API"""
    try:
        if hasattr(image_file, 'seek'):
            image_file.seek(0)
            
        image_data = image_file.read()
        
        headers = {
            'Content-Type': 'application/octet-stream',
            'Content-Disposition': f'attachment; filename="{image_file.name}"'
        }
        
        response = requests.post(
            f"{REST_API_URL}/search/image",
            data=image_data,
            headers=headers,
            timeout=30
        )
        response.raise_for_status()
        
        results = response.json()
        if not isinstance(results, list):
            results = [results] 
            
        return [Product(item) for item in results]
    except Exception as e:
        st.error(f"Error searching by image: {str(e)}")
        if 'response' in locals() and hasattr(response, 'text'):
            st.error(f"Response: {response.text}")
        return []

def search_by_description_graphql(description: str) -> List[Product]:
    """Search products by description using GraphQL"""
    try:
        query = """
        mutation SearchProducts($description: String!) {
            search(description: $description) {
                id
                s3Path
                productDisplayName
                masterCategory
                subCategory
                articleType
                baseColour
                season
                gender
                year
            }
        }
        """
        response = requests.post(
            GRAPHQL_URL,
            json={
                "query": query,
                "variables": {"description": description}
            },
            headers={"Content-Type": "application/json"},
            timeout=10
        )
        response.raise_for_status()
        data = response.json()
        
        # Handle potential GraphQL errors
        if "errors" in data:
            error_messages = [e.get("message", "Unknown error") for e in data["errors"]]
            st.error("GraphQL Errors: " + ", ".join(error_messages))
            return []
            
        results = data.get("data", {}).get("search", [])
        if not isinstance(results, list):
            results = [results]  # Ensure we always return a list
            
        return [Product({
            "id": item.get("id"),
            "s3_path": item.get("s3Path"),  # Map s3Path to s3_path
            "productDisplayName": item.get("productDisplayName"),
            "masterCategory": item.get("masterCategory"),
            "subCategory": item.get("subCategory"),
            "articleType": item.get("articleType"),
            "baseColour": item.get("baseColour"),
            "season": item.get("season"),
            "gender": item.get("gender"),
            "year": item.get("year")
        }) for item in results]
        
    except Exception as e:
        st.error(f"Error with GraphQL search: {str(e)}")
        if 'response' in locals():
            st.error(f"Response: {response.text}")
        return []

def display_product_card(product: Product):
    """Display a single product card with product details"""
    with st.container():
        col1, col2 = st.columns([1, 2])
        
        st.sidebar.write("---")
        st.sidebar.subheader("Debug Info")
        st.sidebar.json({
            "s3_path": product.s3_path,
            "image_url": product.image_url,
            "product_name": product.product_display_name
        })
        
        with col1:
            if not product.image_url or not product.image_url.strip():
                st.warning(f"No hay URL de imagen disponible. s3_path: {product.s3_path}")
                return

            client = boto3.client('s3', aws_access_key_id=S3_ACCESS_KEY, aws_secret_access_key=S3_SECRET_KEY,
                                  endpoint_url=S3_ENDPOINT)
            bucket, key = product.s3_path.split('/', 1)
            image_bytes = io.BytesIO()
            client.download_fileobj(bucket, key, image_bytes)
            image_bytes.seek(0)
            image = Image.open(image_bytes).convert('RGB')
            st.image(image, caption=product.product_display_name or "Product Image", use_container_width=True)
                
        with col2:
            if product.product_display_name:
                st.subheader(product.product_display_name)
            
            if any([product.master_category, product.sub_category, product.article_type]):
                category_parts = [
                    product.master_category or "",
                    product.sub_category or "",
                    product.article_type or ""
                ]
                category_str = " » ".join(filter(None, category_parts))
                if category_str:
                    st.caption(f"**Category:** {category_str}")
            
            if product.base_color:
                st.caption(f"**Color:** {product.base_color}")
            
            if product.season:
                st.caption(f"**Season:** {product.season}")
            
            if product.gender:
                st.caption(f"**Gender:** {product.gender}")
            
            if product.year:
                st.caption(f"**Year:** {product.year}")
            
            if product.id:
                st.caption(f"*ID: {product.id}*")
    
    st.divider()

def main():
    st.title("👗 Fashion Product Search")
    st.markdown("Search for fashion products using text descriptions or images with CLIP-based search.")
    
    with st.sidebar:
        st.header("Settings")
        
        api_type = st.radio(
            "Select API Type",
            ["REST", "GraphQL"],
            index=0,
            help="Choose between REST API or GraphQL"
        )
        
        search_mode = st.radio(
            "Search Mode",
            ["Text", "Image"],
            index=0,
            help="Search by text description or upload an image"
        )
        
        if api_type == "GraphQL" and search_mode == "Image":
            st.warning("Image search is only available with the REST API.")
            search_mode = "Text" 
        
        st.markdown("---")
        st.markdown("### API Settings")
        
        global REST_API_URL, GRAPHQL_URL
        if api_type == "REST":
            REST_API_URL = st.text_input("REST API URL", value=REST_API_URL)
        else:
            GRAPHQL_URL = st.text_input("GraphQL URL", value=GRAPHQL_URL)
            
        st.markdown("---")
        st.markdown("### Help")
        st.markdown("""
        - **REST API**: Supports both text and image searches
        - **GraphQL**: Supports text searches only
        """)
    
    st.header("Search")
    
    st.markdown("""
    <style>
        .stApp {
            max-width: 1200px;
            margin: 0 auto;
        }
        .stButton>button {
            width: 100%;
            border-radius: 20px;
            background-color: #4CAF50;
            color: white;
            font-weight: bold;
        }
        .stTextInput>div>div>input {
            border-radius: 20px;
            padding: 10px;
        }
        .stFileUploader {
            border: 2px dashed #ccc;
            border-radius: 20px;
            padding: 20px;
            text-align: center;
        }
        .stAlert {
            border-radius: 10px;
        }
        .stSpinner > div > div {
            border-color: #4CAF50 transparent transparent transparent;
        }
        .stMarkdown h3 {
            color: #4CAF50;
        }
    </style>
    """, unsafe_allow_html=True)
    
    if search_mode == "Text":
        with st.form("text_search_form"):
            col1, col2 = st.columns([4, 1])
            with col1:
                search_query = st.text_input(
                    "Describe the fashion item you're looking for:",
                    placeholder="e.g., red summer dress with floral pattern",
                    label_visibility="collapsed"
                )
            with col2:
                search_button = st.form_submit_button("🔍 Search", type="primary")
        
        if search_button:
            if not search_query.strip():
                st.warning("Please enter a search query.")
            else:
                with st.spinner("🔍 Searching for products..."):
                    try:
                        if api_type == "REST":
                            products = search_by_description_rest(search_query)
                        else:  # GraphQL
                            products = search_by_description_graphql(search_query)
                        
                        if products:
                            st.success(f"✅ Found {len(products)} products matching your search")
                            for product in products:
                                display_product_card(product)
                        else:
                            st.warning("No products found matching your query. Try different keywords.")
                    except Exception as e:
                        st.error(f"❌ An error occurred during search: {str(e)}")
                        st.exception(e)  # Show full traceback in debug mode
    
    # Search by image
    else:
        st.info("ℹ️ Upload an image to find visually similar fashion products")
        uploaded_file = st.file_uploader(
            "",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=False,
            label_visibility="collapsed"
        )
        
        if uploaded_file is not None:
            if api_type == "REST":
                with st.spinner("🔍 Analyzing image and searching for similar products..."):
                    try:
                        products = search_by_image_rest(uploaded_file)
                        
                        if products:
                            st.success(f"✅ Found {len(products)} similar products")
                            for product in products:
                                display_product_card(product)
                        else:
                            st.warning("No similar products found. Try a different image.")
                    except Exception as e:
                        st.error(f"❌ An error occurred during image search: {str(e)}")
                        st.exception(e)
            else:
                st.warning("⚠️ Image search is only available with the REST API. Please switch to REST API in the sidebar.")
    
if __name__ == "__main__":
    main()
