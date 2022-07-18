# Web scraping scripts

## Amazon web scraping scripts

### Motivation  
Extracting
* product information, descriptions, price, rating  
* product reviews

### Perfomance considerations taken into account:
Default html parser in BeautifulSoup is written in Python and exhibits slow performance that most likely will be troublesome in scaling. There small changes that make BeautifulSoup faster: 
* Using lxml - an XML and HTML parser https://lxml.de/
* Using cchardet - an universal character encoding detector
* Creating sessions for requests

Source: https://beautiful-soup-4.readthedocs.io/en/latest/#improving-performance 

### Further improvements to consider:  
* Multithreading
* Running as persistent session  
* handling dynamic/JavaScript websites

## (On-going) Metaculus web crawling and scraping   

### Motivation
Extracting:  
* questions and predictions data
* comments

Using docker+splash v3.5 with Scrapy


