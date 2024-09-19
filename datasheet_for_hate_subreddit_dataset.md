# Datasheet for dataset of hate subreddits

Questions from the [Datasheets for Datasets](https://arxiv.org/abs/1803.09010) paper, v7.

Jump to section:

- [Motivation](#motivation)
- [Composition](#composition)
- [Collection process](#collection-process)
- [Preprocessing/cleaning/labeling](#preprocessingcleaninglabeling)
- [Uses](#uses)
- [Distribution](#distribution)
- [Maintenance](#maintenance)

## Motivation

_The questions in this section are primarily intended to encourage dataset creators
to clearly articulate their reasons for creating the dataset and to promote transparency
about funding interests._

### For what purpose was the dataset created? 

This dataset was created to obtain a representative set of hate subreddits that could be used to study their dynamics -- namely, this dataset was used to determine the characteristics of users who participate in multiple types of hate subreddits.

### Who created the dataset (e.g., which team, research group) and on behalf of which entity (e.g., company, institution, organization)?
Omitted for anonymity.

### Who funded the creation of the dataset? 
Omitted for anonymity.

### Any other comments?

## Composition

_Most of these questions are intended to provide dataset consumers with the
information they need to make informed decisions about using the dataset for
specific tasks. The answers to some of these questions reveal information
about compliance with the EU’s General Data Protection Regulation (GDPR) or
comparable regulations in other jurisdictions._

### What do the instances that comprise the dataset represent (e.g., documents, photos, people, countries)?

Each instance of the dataset represents a subreddit.

### How many instances are there in total (of each type, if appropriate)?

There are 168 instances: 15 racist subreddits, 16 anti-LGBTQ, 32 misogynistic, 58 general hate, 7 Islamophobic, 35 xenophobic, 4 antisemitic, and 1 ableist.

### Does the dataset contain all possible instances or is it a sample (not necessarily random) of instances from a larger set?

The dataset is a sample of subreddits from a larger collection of crowdsourced lists of subreddits that are banned or deemed as hate subreddits.

### What data does each instance consist of? 

Each instance consists of the name of a subreddit, the category of the subreddit, and standardized scores across a range of targeted identities, indicating how much hate speech directed at each identity is present in the subreddit relative to 

### Is there a label or target associated with each instance?

Each subreddit is labeled as either "racist," "anti-LGBTQ," "misogynistic," "general hate", "xenophobic," "Islamophobic," "antisemitic," or "ableist."

### Is any information missing from individual instances?

No.

### Are relationships between individual instances made explicit (e.g., users’ movie ratings, social network links)?

Relations among subreddits are not defined in this case.

### Are there recommended data splits (e.g., training, development/validation, testing)?

N/A.

### Are there any errors, sources of noise, or redundancies in the dataset?

The model used to label hate speech in the subreddits is imperfect, and thus a source of noise. The R-squared value of the number of posts detected by it ranges from 0.83 for racist posts to 0.09 for transphobic posts.

### Is the dataset self-contained, or does it link to or otherwise rely on external resources (e.g., websites, tweets, other datasets)?

The dataset is most useful when combined with the data from the subreddits, which can be obtained from Pushshift.

### Does the dataset contain data that might be considered confidential (e.g., data that is protected by legal privilege or by doctor-patient confidentiality, data that includes the content of individuals’ non-public communications)?

No.

### Does the dataset contain data that, if viewed directly, might be offensive, insulting, threatening, or might otherwise cause anxiety?

Yes. Several of the subreddit names contain slurs or offensive phrases directed at marginalized groups.

### Does the dataset relate to people? 

Yes.

### Does the dataset identify any subpopulations (e.g., by age, gender)?

No.

### Is it possible to identify individuals (i.e., one or more natural persons), either directly or indirectly (i.e., in combination with other data) from the dataset?

No.

### Does the dataset contain data that might be considered sensitive in any way (e.g., data that reveals racial or ethnic origins, sexual orientations, religious beliefs, political opinions or union memberships, or locations; financial or health data; biometric or genetic data; forms of government identification, such as social security numbers; criminal history)?

No.

### Any other comments?

## Collection process

_\[T\]he answers to questions here may provide information that allow others to
reconstruct the dataset without access to it._

### How was the data associated with each instance acquired?

Comments and submissions from each subreddit were randomly sampled from the Pushshift dataset (Baumgartner et al.). The number of comments and submissions containing hate speech of different categories was predicted using a deep-learning model. 

### What mechanisms or procedures were used to collect the data (e.g., hardware apparatus or sensor, manual human curation, software program, software API)?

For each subreddit, a sample of comments and submissions was collected, and a deep-learning model was used to detect the number of comments and submissions containing hate speech of each category. See paper for more details on hate speech detection. These predictions were validated on a subset of 10 submissions and comments from a random sample of 25 subreddits. Ground truth labels were obtained from trained annotators who counted the number of submissions and comments containing each type of hate speech in each subreddit. Across all categories, the average R^{2} value of the hate speech prediction method is 0.49.

### If the dataset is a sample from a larger set, what was the sampling strategy (e.g., deterministic, probabilistic with specific sampling probabilities)?

Subreddits with fewer than 1000 users were excluded from analysis, as these represent less than 2% of all subreddits considered. Additionally, subreddits deemed by a human annotator to be clearly irrelevant to hate were excluded.

### Who was involved in the data collection process (e.g., students, crowdworkers, contractors) and how were they compensated (e.g., how much were crowdworkers paid)?

Undergraduate research assistants were involved in the collection of data and validation of the model via manual annotations.

### Over what timeframe was the data collected?

The data were collected between October 2023 and April 2024. However, the data represent subreddits active as early as 2008.

### Were any ethical review processes conducted (e.g., by an institutional review board)?

No.

### Does the dataset relate to people?

The dataset does not represent specific individuals.

### Did you collect the data from the individuals in question directly, or obtain it via third parties or other sources (e.g., websites)?

Comments produced by Reddit users were obtained from the Pushshift API (Baumgartner et al.).

### Were the individuals in question notified about the data collection?

No.

### Did the individuals in question consent to the collection and use of their data?

The individuals in question did not consent to this specific use case, though we only used publicly available information as data.

### If consent was obtained, were the consenting individuals provided with a mechanism to revoke their consent in the future or for certain uses?

N/A.

### Has an analysis of the potential impact of the dataset and its use on data subjects (e.g., a data protection impact analysis) been conducted?

No.

### Any other comments?

## Preprocessing/cleaning/labeling

_The questions in this section are intended to provide dataset consumers with the information
they need to determine whether the “raw” data has been processed in ways that are compatible
with their chosen tasks. For example, text that has been converted into a “bag-of-words” is
not suitable for tasks involving word order._

### Was any preprocessing/cleaning/labeling of the data done (e.g., discretization or bucketing, tokenization, part-of-speech tagging, SIFT feature extraction, removal of instances, processing of missing values)?

Hate speech labels were assigned to groups of Reddit posts using deep learning methods.

### Was the “raw” data saved in addition to the preprocessed/cleaned/labeled data (e.g., to support unanticipated future uses)?

Yes, raw Reddit text is available upon motivated request.

### Is the software used to preprocess/clean/label the instances available?

Yes, the code used to train the hate speech detection model is in the same GitHub repository.

### Any other comments?

## Uses

_These questions are intended to encourage dataset creators to reflect on the tasks
for which the dataset should and should not be used. By explicitly highlighting these tasks,
dataset creators can help dataset consumers to make informed decisions, thereby avoiding
potential risks or harms._

### Has the dataset been used for any tasks already?

Yes. The dataset has been used to conduct an analysis characterizing how users move among different categories of hate subreddits.

### Is there a repository that links to any or all papers or systems that use the dataset?

No.

### What (other) tasks could the dataset be used for?

The dataset could be used for an abundance of other tasks related to how to reduce hate in online environments or understanding how users become radicalized online. For example, several studies have explored the impact of hate subreddit bans or quarantines (Chandrasekharan et al., 2017; Chandrasekharan et al., 2022), or the impact of interactions with members of hate subreddits (Russo et al., 2024). However, these studies only focus on a few hate subreddits. These studies can be replicated using this dataset to understand if their results generalize to the larger set of hate subreddits.

### Is there anything about the composition of the dataset or the way it was collected and preprocessed/cleaned/labeled that might impact future uses?

No.

### Are there tasks for which the dataset should not be used?

The data from these hate subreddits should not be used to train models that generate text, as they will likely generate offensive speech.

### Any other comments?

## Distribution

### Will the dataset be distributed to third parties outside of the entity (e.g., company, institution, organization) on behalf of which the dataset was created? 

The dataset will be publicly available for anyone to use.

### How will the dataset will be distributed (e.g., tarball on website, API, GitHub)?

The dataset will be distributed via GitHub.

### When will the dataset be distributed?

The dataset will be distributed upon acceptance of the paper that accompanies it.

### Will the dataset be distributed under a copyright or other intellectual property (IP) license, and/or under applicable terms of use (ToU)?

No.

### Have any third parties imposed IP-based or other restrictions on the data associated with the instances?

No.

### Do any export controls or other regulatory restrictions apply to the dataset or to individual instances?

No.

### Any other comments?

## Maintenance

_These questions are intended to encourage dataset creators to plan for dataset maintenance
and communicate this plan with dataset consumers._

### Who is supporting/hosting/maintaining the dataset?

Omitted for anonymity.

### How can the owner/curator/manager of the dataset be contacted (e.g., email address)?

Omitted for anonymity.

### Is there an erratum?

No.

### Will the dataset be updated (e.g., to correct labeling errors, add new instances, delete instances)?

If any issues are found with the dataset, it will be updated.

### If the dataset relates to people, are there applicable limits on the retention of the data associated with the instances (e.g., were individuals in question told that their data would be retained for a fixed period of time and then deleted)?

N/A.

### Will older versions of the dataset continue to be supported/hosted/maintained?

Yes, older versions will be available on GitHub.

### If others want to extend/augment/build on/contribute to the dataset, is there a mechanism for them to do so?

If others wish to build on the dataset and host it on the GitHub the dataset is hosted on, they may contact the authors and ask them to do so. Alternatively, they can host it on their own GitHub account.

### Any other comments?

### References

Baumgartner, Jason, et al. "The pushshift reddit dataset." Proceedings of the international AAAI conference on web and social media. Vol. 14. 2020.

Chandrasekharan, Eshwar, et al. "You can't stay here: The efficacy of reddit's 2015 ban examined through hate speech." Proceedings of the ACM on human-computer interaction 1.CSCW (2017): 1-22.

Chandrasekharan, Eshwar, et al. "Quarantined! Examining the effects of a community-wide moderation intervention on Reddit." ACM Transactions on Computer-Human Interaction (TOCHI) 29.4 (2022): 1-26.

Russo, Giuseppe, Manoel Horta Ribeiro, and Robert West. "Stranger Danger! Cross-Community Interactions with Fringe Users Increase the Growth of Fringe Communities on Reddit." arXiv preprint arXiv:2310.12186 (2023).
