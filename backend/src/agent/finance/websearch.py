import trafilatura

CANADA_PROPERTY_TAX_RATE_URL="https://catax.tools/property-tax-by-city/"
CANADA_PROPERTY_TAX_RATE_DEFAULT=0.015

def _fetch_link_content(url:str) -> str:
    downloaded = trafilatura.fetch_url(url)
    content = trafilatura.extract(
        downloaded,
        output_format="markdown",
        include_links=True,
        include_tables=True,
        include_images=False,
    )
    return content