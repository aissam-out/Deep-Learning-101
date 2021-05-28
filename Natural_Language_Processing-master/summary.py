# Import resources
from summa.summarizer import summarize
from summa import keywords

# Init summarizer parameters
title = "first text"
text = """Automatic summarization is the process of reducing a text document with a \
computer program in order to create a summary that retains the most important points \
of the original document. As the problem of information overload has grown, and as \
the quantity of data has increased, so has interest in automatic summarization. \
Technologies that can make a coherent summary take into account variables such as \
length, writing style and syntax. An example of the use of summarization technology \
is search engines such as Google. Document summarization is another."""

print(summarize(text, ratio=0.2))

print(summarize(text, words=20))

print(keywords.keywords(text))
