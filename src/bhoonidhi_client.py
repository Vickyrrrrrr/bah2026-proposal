import os
import requests
from pathlib import Path
from pystac_client import Client

class BhoonidhiClient:
    """
    Programmatic interface for the ISRO NRSC Bhoonidhi STAC API.
    Handles JWT authentication, spatial-temporal searches, and downloading GeoTIFFs.
    """
    def __init__(self, username=None, password=None, base_url="https://bhoonidhi-api.nrsc.gov.in"):
        self.username = username or os.getenv("BHOO_USERNAME")
        self.password = password or os.getenv("BHOO_PASSWORD")
        self.base_url = base_url.rstrip('/')
        self.token = None
        self.headers = {}
        
        if self.username and self.password:
            self.authenticate()

    def authenticate(self):
        """Request a JWT Bearer token from the auth endpoint."""
        auth_url = f"{self.base_url}/auth/token"
        payload = {
            "userId": self.username,
            "password": self.password,
            "grant_type": "password"
        }
        try:
            response = requests.post(auth_url, json=payload, timeout=15)
            if response.status_code == 200:
                self.token = response.json().get("access_token")
                self.headers = {"Authorization": f"Bearer {self.token}"}
                print("✅ Successfully authenticated with Bhoonidhi STAC API.")
            else:
                print(f"❌ Authentication failed (status code: {response.status_code})")
                print(response.text)
        except Exception as e:
            print(f"❌ Failed to connect to auth server: {e}")

    def search_scenes(self, collection, bbox, date_range, cloud_cover_gt=30):
        """
        Query the STAC catalog for matching satellite products.
        """
        if not self.token:
            print("⚠️ Not authenticated. Searching as anonymous user (if supported).")
            
        stac_url = f"{self.base_url}/data"
        try:
            client = Client.open(stac_url, headers=self.headers)
            search = client.search(
                collections=[collection],
                bbox=bbox,
                datetime=date_range,
                query={"eo:cloud_cover": {"gt": cloud_cover_gt}}
            )
            items = list(search.get_items())
            print(f"🔍 Found {len(items)} matching items in collection '{collection}'.")
            return items
        except Exception as e:
            print(f"❌ Error querying STAC API: {e}")
            return []

    def download_asset(self, item, out_dir):
        """Download a specific asset (GeoTIFF) from a STAC item using direct download endpoint."""
        if not self.headers:
            print("⚠️ Auth headers missing. Download might fail.")
            
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        
        # Construct the official download URL
        collection_id = getattr(item, 'collection_id', None) or item.properties.get('collection') or 'Resourcesat2_LISS4'
        url = f"{self.base_url}/download?id={item.id}&collection={collection_id}"
        filename = out_path / f"{item.id}.tif"
        
        if filename.exists():
            print(f"  File already exists: {filename.name}")
            return filename
            
        print(f"  Downloading {filename.name} from Bhoonidhi direct download endpoint...")
        try:
            with requests.get(url, headers=self.headers, stream=True, timeout=60) as r:
                r.raise_for_status()
                with open(filename, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=16384):
                        f.write(chunk)
            print(f"  ✅ Downloaded {filename.name} successfully.")
            return filename
        except Exception as e:
            print(f"  ❌ Failed to download {filename.name}: {e}")
            return None

if __name__ == "__main__":
    # Example dry-run execution
    client = BhoonidhiClient()
    print("Bhoonidhi STAC Client initialized. Set BHOO_USERNAME and BHOO_PASSWORD to authenticate.")
