# Gutenberg-Data-Analysis
 
In metadata.csv, you’ll find book IDs and book-level attributes, including download counts. In KLDscores.csv, you’ll find book IDs and corresponding “narrative revelation” scores, formatted in the form of Python lists. These scores measure the amount of “information revelation”, measured in terms of Kullback-Liebler divergence, for each subsequent section of the corresponding book. For details on how this measure was constructed, see https://ceur-ws.org/Vol-3558/paper6166.pdf

Steps of analysis performed: 

Build book-level measures of the characteristics of the Kullback-Liebler divergence. Like the average, the variance, the slope of a linear regression across the course of the narrative, etc.
 
Relate these book-level measures of KLD to book popularity by regressing them against log(downloads) at the book level.

Investigate heterogeneity of effects across genres and use LASSO to infer which variables are most independently predictive of log(downloads).
