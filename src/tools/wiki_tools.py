"""
Tools for crawl-related functions
"""

import requests
import xmltodict
import mwapi

def get_pageid_from_revid(rev_id: str, lang='en') -> str:
    session = mwapi.Session(f"https://en.wikipedia.org", user_agent='isaacj@wikimedia.org; PAWS edittypes')
    response = session.get(action="query", prop="revisions", revids=rev_id, rvprop="ids|timestamp|content", rvslots="*")
    page_id = list(response['query']['pages'].values())[0]["pageid"]

    return page_id

def get_wikitext_from_revid(rev_id: str) -> str:
    """get wikitexts from revision id

    Args:
        rev_id (str): revision id of the wikipedia page
    """
    xml_res = requests.get(f"https://en.wikipedia.org/w/api.php?format=xml&action=query&revids={rev_id}&prop=revisions&rvprop=content")  
    dict_res = xmltodict.parse(xml_res.text)
    wikitext = ""
    if "pages" in dict_res["api"]["query"]:
        wikitext = dict_res["api"]["query"]["pages"]["page"]["revisions"]["rev"]["#text"]
    
    return wikitext

def get_prev_cur_wikitext_from_revid_pageid(revid, pageid, lang='en'):
    session = mwapi.Session(f'https://{lang}.wikipedia.org', user_agent='isaacj@wikimedia.org; PAWS edittypes')

    # generate wikitext for revision and previous
    # https://en.wikipedia.org/w/api.php?action=query&prop=revisions&titles=Eve%20Ewing&rvlimit=2&rvdir=older&rvstartid=979988715&rvprop=ids|content|comment&format=json&formatversion=2&rvslots=*
    result = session.get(
        action="query",
        prop="revisions",
        pageids=pageid,
        rvlimit=2,
        rvdir="older",
        rvstartid=revid,
        rvprop="ids|content|comment",
        rvslots="*",
        format='json',
        formatversion=2,
    )
    try:
        curr_wikitext = result['query']['pages'][0]['revisions'][0]['slots']['main']['content']
    except KeyError:
        print("key error, return empty text")
        curr_wikitext = ""  # this shouldn't happen but...
    try:
        prev_wikitext = result['query']['pages'][0]['revisions'][1]['slots']['main']['content']
    except KeyError:
        print("key error, return empty text")
        prev_wikitext = ""  # current revision probaby is first page revision
    return prev_wikitext, curr_wikitext

if __name__ == '__main__':
    from src.tools.wiki_tools import get_wikitext_from_revid
