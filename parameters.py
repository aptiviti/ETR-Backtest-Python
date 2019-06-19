# Parameters

# Import File paths: please insert file paths between quotes.

source_file = "INSERT_CLIENT_FILEPATH"  # ETRSource_Source
spReturns_file = "INSERT_CLIENT_FILEPATH"  # ETRInsight_StockReturns
ftecReturns_file = "INSERT_CLIENT_FILEPATH"  # ETRInsight_FTECReturns

# Adoption Rating, Increase Rating, Decrease Rating, Replacing Rating, Net Score Rating, Market Share Rating parameters
mincitations = 30	 # Minimum citations to be considered for a rating
vcutoff = 1.000  	 # Z-Score value cutoff
dcutoff = 0.675  	 # Delta cutoff

# Peer Rating parameters
peermincitations = 10  # Minimum citations to be consiered for a rating
deltayoy = .05  # Difference between Net Score in this survey vs. a year ago
peerdelta = 2  # Difference between accelerating peers and decelerating peers

# Model Creation parameters
zcutoff = .253  # Minimum difference from 0 in z-score for a label of "Positive" or "Negative"
upperpcutoff = .55  # Minimum p value to be considered "Positive"
lowerpcutoff = .45  # Maximum p value to be considered "Negative"
