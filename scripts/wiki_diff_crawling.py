import subprocess
import pandas as pd
import pdfkit
from bs4 import BeautifulSoup

url_file = "~/Downloads/human-eval-test-sample - human-eval-test-sample.csv"

df = pd.read_csv(url_file)
revids = df['revision_id'].values
urls = ['https://en.wikipedia.org/w/index.php?diff=' + str(revid) + ".html" for revid in revids]

for revid, url in zip(revids, urls):
	wget_command = [
	    "wget",
	    "-p",
	    "--convert-links",
	    url
	]

	try:
	    subprocess.run(wget_command, check=True)
	    print("Download successful.")
	except subprocess.CalledProcessError as e:
	    print(f"Error downloading: {e}")

	html_file_path = "en.wikipedia.org/w/index.php?diff=" + str(revid) + ".html"
	with open(html_file_path, "r", encoding="utf-8") as file:
		html = file.read()

	# Parse the HTML content
	soup = BeautifulSoup(html, "html.parser")

	element_to_remove = soup.find(class_="diff-title")

	if element_to_remove:
	    element_to_remove.extract()


	# Save the modified HTML
	with open(html_file_path, "w", encoding="utf-8") as file:
	    file.write(str(soup))

	save_path = "wiki_pdfs/" + str(revid) + ".pdf"
	try:
		pdfkit.from_file(html_file_path, save_path, options={"enable-local-file-access": "", "load-error-handling":"ignore", "load-media-error-handling":"ignore"})
	except:
		print(f"Error thrown for revid {revid}")