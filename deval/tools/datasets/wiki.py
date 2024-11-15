# The MIT License (MIT)
# Copyright © 2024 Yuma Rao
# Copyright © 2023 Opentensor Foundation

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import re
import sys
import random
import datetime
import bittensor as bt
import wikipedia as wiki
from typing import Dict, List
from queue import Queue, Full
from functools import lru_cache
from .base import Dataset
from ..selector import Selector


# Create a queue called CACHED_ARTICLES to store wikipedia articles that have been fetched
CACHED_ARTICLES = Queue(maxsize=300)

# speed up page loading
@lru_cache(maxsize=1000)
def _get_page(
    title, pageid=None, auto_suggest=False, redirect=True, seed=None
) -> wiki.WikipediaPage:
    """Cached Wikipedia page loading."""
    try:
        page = wiki.page(
            title=title, pageid=pageid, auto_suggest=auto_suggest, redirect=redirect
        )
        # create sections manually if not found
        if not page.sections:
            page._sections = [
                line.strip("= ")
                for line in page.content.splitlines()
                if re.search(r"=+\s+.*\s+=+", line)
            ]
        return page

    except wiki.DisambiguationError as e:
        bt.logging.debug(f"{e.__class__.__name__} loading page {title!r}: {e}")
        # exc info contains a tuple of (requested_title: str, possible_matches: List[str])
        pages = sys.exc_info()[1].args[1]
        if not type(pages) == list:
            return None
        title = random.Random(seed).choice(pages)
        return _get_page(title, auto_suggest=auto_suggest, redirect=redirect)

    except wiki.PageError as e:
        bt.logging.warning(f"{e.__class__.__name__} loading page {title!r}: {e}")
        if not auto_suggest:
            return _get_page(title, auto_suggest=True, redirect=redirect)
        return None


@lru_cache(maxsize=1000)
def _get_random_titles(pages=10, seed=42) -> List:
    """Cached wikipedia random page. Approximately deterministic random titles. This is useful for testing.
    NOTE: the actually cached result will change each session, but the result will be the same within a session.
    """
    return wiki.random(pages=pages)


@lru_cache(maxsize=1000)
def _wiki_search(name, results) -> List:
    """Cached Wikipedia search."""
    return wiki.search(name, results=results)

def process_page(
    page, valid_header: callable = None, valid_content: callable = None
) -> Dict:
    """Process a Wikipedia page and return a dictionary of sections with their content.

    Args:
        page: wikipedia.WikipediaPage
        valid_header: callable to determine if a section header is valid
        valid_content: callable to determine if a section content is valid
    Returns:
        dict: dictionary of sections and their content. Note that keys are tuples (header, section_title)
    """
    header = ""
    sections = {}

    for section_title in page.sections:
        content = page.section(section_title)
        if not content:
            header = section_title
            continue

        # Filter out sections that don't match the headers and/or are not valid
        if (valid_header and not valid_header(header)) or (
            valid_content and not valid_content(content)
        ):
            continue

        key = (header, section_title)
        sections[key] = content.splitlines()

    if not sections:
        sections['full_content'] = [page.content]
        bt.logging.info(f"No valid sections found in page {page.title!r} ({page.url})")

    return sections


def most_relevant_links(page, num_links=10, num_summary_words=50, return_scores=False):
    """Return the most relevant links to a Wikipedia page based on the intersection over union (IOU) of the link and the page summary."""
    link_scores = {}
    summary_words = set(page.summary.split()[:num_summary_words])
    for link in page.links:
        link_words = set(link.split())
        iou = len(summary_words.intersection(link_words)) / len(
            summary_words.union(link_words)
        )
        link_scores[link] = iou / len(link.split())

    sorted_links = sorted(link_scores.items(), key=lambda x: x[1], reverse=True)
    if return_scores:
        return sorted_links[:num_links]

    return [link for link, _ in sorted_links[:num_links]]


def filter_categories(categories, exclude=None, include=None):
    """Filter categories based on a list of categories to exclude and/or include."""
    if exclude:
        categories = [
            cat
            for cat in categories
            if not re.search("|".join(exclude), cat, re.IGNORECASE)
        ]
    if include:
        categories = [
            cat
            for cat in categories
            if re.search("|".join(include), cat, re.IGNORECASE)
        ]
    return categories


class WikiDataset(Dataset):
    """Wikipedia dataset. Uses the wikipedia python api to fetch articles and sections."""
    name = "wiki"
    EXCLUDE_HEADERS = ("See also", "References", "Further reading", "External links")
    EXCLUDE_CATEGORIES = ("articles", "wiki", "pages", "cs1")

    def __init__(
        self,
        min_length_words: int = 50,
        max_links: int = 10,
    ):
        """
        Args:
            min_length_words (int, optional): Minimum section length. Defaults to 50.
            max_links (int, optional): _description_. Defaults to 10.
        """
        self.min_length_words = min_length_words
        self.max_links = max_links

    def get(
        self,
        name: str,
        include: List = None,
        exclude: List = None,
        **kwargs,
    ) -> Dict:
        """Get a specified Wikipedia page and extract a section based on the selector.

        Args:
            name (_type_): _description_
            pageid (_type_, optional): _description_. Defaults to None.
            auto_suggest (bool, optional): _description_. Defaults to True.
            redirect (bool, optional): _description_. Defaults to True.
            include (List, optional): _description_. Defaults to None.
            exclude (List, optional): _description_. Defaults to None.

        Returns:
            Dict: _description_
        """

        page = _get_page(title=name, **kwargs)
        if page is None:
            return None

        # Only return a sections with a minimum number of words
        exclude = (exclude or []) + list(self.EXCLUDE_HEADERS)
        sections = process_page(
            page,
            valid_header=lambda x: x not in exclude and (not include or x in include),
            valid_content=lambda x: len(x.split()) >= self.min_length_words,
        )
        if not sections:
            return None
        
        topic = "All Sections"
        content = "\n".join(["\n".join(s) for _, s in sections.items()])
        section_length = len(content.split())
        section_title = None

        context = {
            "title": name,  # title of wiki article
            "topic": topic,  # title of wiki section
            "subtopic": section_title,
            "content": content,
            "sections": sections,
            "internal_links": list(filter(lambda x: x not in exclude, page.sections)),
            "external_links": most_relevant_links(page, num_links=self.max_links),
            "tags": filter_categories(page.categories, exclude=self.EXCLUDE_CATEGORIES),
            "source": "Wikipedia",
            "extra": {
                "url": page.url,
                "page_length": len(page.content.split()),
                "section_length": section_length,
            },
        }
        try:
            CACHED_ARTICLES.put(context, block=False)
        except Full:
            bt.logging.debug("Cache is full. Skipping article until cache is emptied.")
        return context

    def search(self, name, results=3, selector: Selector = None) -> Dict:
        titles = _wiki_search(name, results=results)
        title = selector(titles)
        return self.get(title)

    def random(self, pages=10, seed=None, selector: Selector = None, **kwargs) -> Dict:
        titles = (
            wiki.random(pages=pages)
            if seed is None
            else _get_random_titles(pages=pages, seed=seed)
        )
        title = selector(titles)
        return self.get(title)


