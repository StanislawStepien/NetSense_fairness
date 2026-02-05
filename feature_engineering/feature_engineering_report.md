# Feature Engineering Report

## Pure Demographics
- Rows: 1188
- Columns: 21
- Missing data: none

### Numeric Summary
- computeruse: mean=0.6768, std=0.4786, min=0.0000, p25=0.0000, median=1.0000, p75=1.0000, max=2.0000
- contactlens: mean=0.3838, std=0.4865, min=0.0000, p25=0.0000, median=0.0000, p75=1.0000, max=1.0000
- daded: mean=1.1313, std=1.1477, min=0.0000, p25=0.0000, median=1.0000, p75=2.0000, max=4.0000
- dadrelig: mean=1.4697, std=1.0577, min=0.0000, p25=1.0000, median=1.0000, p75=1.0000, max=4.0000
- disabilitylearning: mean=0.0253, std=0.2118, min=0.0000, p25=0.0000, median=0.0000, p75=0.0000, max=2.0000
- disabilityphysical: mean=0.0051, std=0.0709, min=0.0000, p25=0.0000, median=0.0000, p75=0.0000, max=1.0000
- ethnicity: mean=1.8990, std=0.9432, min=0.0000, p25=2.0000, median=2.0000, p75=2.0000, max=5.0000
- eyeglasses: mean=0.4747, std=0.4996, min=0.0000, p25=0.0000, median=0.0000, p75=1.0000, max=1.0000
- familymilitary: mean=0.4444, std=0.4971, min=0.0000, p25=0.0000, median=0.0000, p75=1.0000, max=1.0000
- fbprivacy: mean=0.4192, std=0.9805, min=0.0000, p25=0.0000, median=0.0000, p75=0.0000, max=4.0000
- gender: mean=0.5303, std=0.4993, min=0.0000, p25=0.0000, median=1.0000, p75=1.0000, max=1.0000
- heighttotal: mean=1.1717, std=0.8595, min=0.0000, p25=1.0000, median=1.0000, p75=2.0000, max=4.0000
- major: mean=2.8434, std=1.8920, min=0.0000, p25=1.0000, median=2.0000, p75=5.0000, max=6.0000
- momed: mean=1.5000, std=1.0042, min=0.0000, p25=0.0000, median=2.0000, p75=2.0000, max=4.0000
- momrelig: mean=1.4242, std=0.9654, min=0.0000, p25=1.0000, median=1.0000, p75=1.0000, max=4.0000
- numberpets: mean=1.5253, std=0.9835, min=0.0000, p25=1.0000, median=2.0000, p75=2.0000, max=3.0000
- parentsmarriage: mean=1.8535, std=0.3943, min=0.0000, p25=2.0000, median=2.0000, p75=2.0000, max=3.0000
- pincome: mean=0.4848, std=0.7370, min=0.0000, p25=0.0000, median=0.0000, p75=1.0000, max=4.0000
- weight: mean=1.0455, std=1.0796, min=0.0000, p25=0.0000, median=1.0000, p75=2.0000, max=3.0000
- SurveyNr: mean=3.5000, std=1.7085, min=1.0000, p25=2.0000, median=3.5000, p75=5.0000, max=6.0000
- egoid: mean=51158.6869, std=24485.2160, min=10060.0000, p25=30076.0000, median=50908.0000, p75=71700.0000, max=99338.0000

### Top Correlations (absolute)
- dadrelig ↔ momrelig: 0.6760
- gender ↔ heighttotal: 0.4708
- contactlens ↔ eyeglasses: 0.3727
- disabilitylearning ↔ disabilityphysical: 0.3280
- dadrelig ↔ parentsmarriage: 0.2955
- daded ↔ parentsmarriage: 0.2591
- daded ↔ momed: 0.2588
- dadrelig ↔ ethnicity: 0.2362
- disabilityphysical ↔ heighttotal: 0.2345
- heighttotal ↔ major: 0.2041

### Outlier Counts (IQR)
- heighttotal: 30
- pincome: 12

## Topology Features
- Rows: 13244
- Columns: 7
- Missing data: none

### Numeric Summary
- EgoID: mean=142548.9553, std=39555.6607, min=10060.0000, p25=118996.0000, median=146154.0000, p75=173551.0000, max=199995.0000
- SurveyNr: mean=3.0870, std=1.6431, min=1.0000, p25=2.0000, median=3.0000, p75=4.0000, max=6.0000
- Degree Centrality: mean=0.0009, std=0.0018, min=0.0003, p25=0.0003, median=0.0004, p75=0.0007, max=0.0170
- Closeness Centrality: mean=0.0259, std=0.0326, min=0.0003, p25=0.0032, median=0.0056, p75=0.0607, max=0.1227
- Betweenness Centrality: mean=0.0009, std=0.0057, min=0.0000, p25=0.0000, median=0.0000, p75=0.0000, max=0.1543
- Eigenvector Centrality: mean=0.0037, std=0.0210, min=0.0000, p25=0.0000, median=0.0000, p75=0.0000, max=0.4668
- Community: mean=38.0803, std=37.7695, min=0.0000, p25=8.0000, median=25.0000, p75=57.0000, max=157.0000

### Top Correlations (absolute)
- EgoID ↔ Degree Centrality: 0.5813
- Degree Centrality ↔ Betweenness Centrality: 0.5089
- Closeness Centrality ↔ Community: 0.5071
- Closeness Centrality ↔ Betweenness Centrality: 0.2840
- EgoID ↔ Betweenness Centrality: 0.2373
- Degree Centrality ↔ Eigenvector Centrality: 0.2371
- Betweenness Centrality ↔ Eigenvector Centrality: 0.2106
- Closeness Centrality ↔ Eigenvector Centrality: 0.1796
- SurveyNr ↔ Community: 0.1782
- Degree Centrality ↔ Closeness Centrality: 0.1579

### Outlier Counts (IQR)
- EgoID: 351
- Degree Centrality: 1141
- Eigenvector Centrality: 2851
- Community: 435

## Merged Features
- Rows: 5168
- Columns: 80
- Missing data: none

### Numeric Summary
- egoid: mean=49771.7713, std=23172.7568, min=10060.0000, p25=30076.0000, median=50181.0000, p75=70889.0000, max=89827.0000
- SurveyNr: mean=3.7767, std=1.3587, min=2.0000, p25=3.0000, median=4.0000, p75=5.0000, max=6.0000
- Discussed_Politics: mean=12.4094, std=4.4479, min=0.0000, p25=8.0000, median=12.0000, p75=16.0000, max=20.0000
- Discussed_Religion: mean=19.0652, std=8.9597, min=0.0000, p25=17.0000, median=17.0000, p75=26.0000, max=34.0000
- Performed_Volunteer_Work: mean=9.4249, std=4.7215, min=0.0000, p25=4.0000, median=8.0000, p75=12.0000, max=20.0000
- Felt_Depressed: mean=9.5224, std=4.5963, min=0.0000, p25=4.0000, median=8.0000, p75=12.0000, max=20.0000
- timeperweekactivea: mean=4.9719, std=3.6842, min=0.0000, p25=0.0000, median=7.0000, p75=8.0000, max=9.0000
- Drank_Beer: mean=9.0480, std=5.0388, min=0.0000, p25=4.0000, median=8.0000, p75=16.0000, max=20.0000
- timeperweekactiveb: mean=22.8311, std=10.0782, min=0.0000, p25=21.0000, median=26.0000, p75=30.0000, max=34.0000
- Exercised: mean=15.4303, std=4.3806, min=0.0000, p25=12.0000, median=16.0000, p75=20.0000, max=20.0000
- Drank_Wine_or_Liquor: mean=6.6393, std=3.7683, min=0.0000, p25=2.0000, median=5.0000, p75=11.0000, max=13.0000
- Felt_Overwhelmed: mean=29.2483, std=8.8231, min=8.0000, p25=25.0000, median=34.0000, p75=34.0000, max=42.0000
- Political_Campaign_Work: mean=0.3433, std=0.2895, min=0.0000, p25=0.2000, median=0.2000, p75=0.2000, max=1.0000
- Public_Communication: mean=3.7430, std=4.9154, min=0.0000, p25=0.0000, median=0.0000, p75=4.0000, max=16.0000
- Religious_Attendance: mean=12.3584, std=4.8746, min=0.0000, p25=8.0000, median=16.0000, p75=16.0000, max=20.0000
- abstinent: mean=0.2434, std=0.4292, min=0.0000, p25=0.0000, median=0.0000, p75=0.0000, max=1.0000
- Potential_College_Transfer_Score: mean=8.3214, std=4.3338, min=0.0000, p25=5.0000, median=5.0000, p75=14.0000, max=16.0000
- Likelihood_Student_Gov_Participation_Score: mean=13.6173, std=5.3291, min=0.0000, p25=8.0000, median=12.0000, p75=17.0000, max=25.0000
- Experiences_of_Depression_Score: mean=21.3849, std=8.7136, min=0.0000, p25=17.0000, median=26.0000, p75=26.0000, max=34.0000
- Future_Academic_Changes_Score: mean=6.7585, std=5.5792, min=0.0000, p25=4.0000, median=4.0000, p75=12.0000, max=16.0000
- Likelihood_Sports_Participation_Score: mean=19.6211, std=5.5268, min=0.0000, p25=17.0000, median=21.0000, p75=25.0000, max=25.0000
- Active_Community_Volunteerism_Score: mean=20.3553, std=8.3623, min=0.0000, p25=17.0000, median=17.0000, p75=26.0000, max=34.0000
- Artistic Achievement_Score: mean=11.7531, std=4.1825, min=0.0000, p25=8.0000, median=12.0000, p75=16.0000, max=20.0000
- Professional Development_Score: mean=20.6720, std=6.4179, min=0.0000, p25=17.0000, median=23.0000, p75=26.0000, max=28.0000
- Family and Personal Life_Score: mean=21.1598, std=7.1125, min=0.0000, p25=17.0000, median=23.0000, p75=26.0000, max=28.0000
- Classical_Traditional_Score: mean=13.4536, std=9.7091, min=0.0000, p25=8.0000, median=17.0000, p75=17.0000, max=42.0000
- Modern_Popular_Score: mean=5.2175, std=4.5311, min=0.0000, p25=0.0000, median=4.0000, p75=8.0000, max=20.0000
- Folk_Cultural_Score: mean=4.1515, std=6.0704, min=0.0000, p25=0.0000, median=0.0000, p75=8.0000, max=25.0000
- Nostalgic_Easy_Score: mean=14.3388, std=10.2301, min=0.0000, p25=8.0000, median=17.0000, p75=17.0000, max=51.0000
- Musical_Theater_Score: mean=3.3019, std=4.5024, min=0.0000, p25=0.0000, median=0.0000, p75=8.0000, max=16.0000
- Music_Diversity_Score: mean=12.6765, std=5.0231, min=0.0000, p25=8.0000, median=12.0000, p75=16.0000, max=20.0000
- Classical_Traditional_Preferred: mean=0.8166, std=0.3871, min=0.0000, p25=1.0000, median=1.0000, p75=1.0000, max=1.0000
- Modern_Popular_Preferred: mean=0.7396, std=0.4389, min=0.0000, p25=0.0000, median=1.0000, p75=1.0000, max=1.0000
- Nostalgic_Easy_Preferred: mean=0.8543, std=0.3528, min=0.0000, p25=1.0000, median=1.0000, p75=1.0000, max=1.0000
- Fiction_Score: mean=6.3469, std=3.8915, min=0.0000, p25=2.0000, median=5.0000, p75=11.0000, max=13.0000
- NonFiction_Score: mean=4.6610, std=5.6844, min=0.0000, p25=0.0000, median=0.0000, p75=8.0000, max=16.0000
- SelfHelp_Informational_Score: mean=3.9985, std=4.7034, min=0.0000, p25=0.0000, median=0.0000, p75=8.0000, max=16.0000
- Reading_Diversity_Score: mean=15.0803, std=6.7859, min=0.0000, p25=8.0000, median=17.0000, p75=17.0000, max=25.0000
- Fiction_Preferred: mean=0.9058, std=0.2922, min=0.0000, p25=1.0000, median=1.0000, p75=1.0000, max=1.0000
- enjoy_movies: mean=3.8036, std=0.4460, min=2.0000, p25=4.0000, median=4.0000, p75=4.0000, max=4.0000
- extraversion: mean=14.4621, std=6.3531, min=0.0000, p25=10.0000, median=14.0000, p75=19.0000, max=28.0000
- abortion: mean=1.6529, std=2.0743, min=-1.0000, p25=0.0000, median=2.0000, p75=3.0000, max=5.0000
- openness: mean=30.8466, std=6.6217, min=0.0000, p25=27.0000, median=31.0000, p75=36.0000, max=47.0000
- premaritalsex: mean=1.9594, std=1.3605, min=0.0000, p25=1.0000, median=2.0000, p75=2.0000, max=4.0000
- euthanasia: mean=2.6885, std=2.2033, min=0.0000, p25=0.0000, median=2.0000, p75=5.0000, max=5.0000
- homosexual: mean=2.2018, std=1.7576, min=0.0000, p25=0.0000, median=2.0000, p75=4.0000, max=4.0000
- deathpen: mean=1.3847, std=1.9137, min=0.0000, p25=0.0000, median=0.0000, p75=2.0000, max=5.0000
- marijuana: mean=2.4356, std=2.2534, min=-1.0000, p25=0.0000, median=2.0000, p75=5.0000, max=5.0000
- eqchances: mean=2.7183, std=1.2928, min=-1.0000, p25=2.0000, median=2.0000, p75=4.0000, max=5.0000
- enjoy_games: mean=2.9125, std=1.0233, min=0.0000, p25=2.0000, median=3.0000, p75=4.0000, max=4.0000
- fssocsec: mean=2.1761, std=0.9911, min=-1.0000, p25=1.0000, median=2.0000, p75=3.0000, max=4.0000
- occupationmom: mean=5.7326, std=2.2170, min=1.0000, p25=6.0000, median=7.0000, p75=7.0000, max=7.0000
- toomucheqrights: mean=2.8522, std=1.1675, min=-1.0000, p25=2.0000, median=3.0000, p75=4.0000, max=5.0000
- happy: mean=3.2902, std=1.4674, min=-1.0000, p25=3.0000, median=3.0000, p75=4.0000, max=5.0000
- agreeableness: mean=16.9977, std=5.6332, min=0.0000, p25=13.0000, median=17.0000, p75=21.0000, max=34.0000
- enjoy_follow_sports: mean=2.9603, std=1.0654, min=0.0000, p25=2.0000, median=3.0000, p75=4.0000, max=4.0000
- conscientiousness: mean=23.8942, std=6.7433, min=0.0000, p25=20.0000, median=24.0000, p75=29.0000, max=41.0000
- neuroticism: mean=17.0341, std=6.5579, min=0.0000, p25=13.0000, median=17.0000, p75=22.0000, max=36.0000
- health: mean=3.1101, std=1.3700, min=1.0000, p25=2.0000, median=2.0000, p75=5.0000, max=5.0000
- reading_category: mean=0.8506, std=0.7078, min=-1.0000, p25=0.0000, median=1.0000, p75=1.0000, max=4.0000
- Degree Centrality_Survey: mean=0.0066, std=0.0035, min=0.0003, p25=0.0043, median=0.0061, p75=0.0087, max=0.0170
- Closeness Centrality_Survey: mean=0.0410, std=0.0383, min=0.0003, p25=0.0059, median=0.0147, p75=0.0788, max=0.1227
- Betweenness Centrality_Survey: mean=0.0105, std=0.0188, min=0.0000, p25=0.0000, median=0.0005, p75=0.0127, max=0.1543
- Eigenvector Centrality_Survey: mean=0.0184, std=0.0670, min=0.0000, p25=0.0000, median=0.0000, p75=0.0011, max=0.4668
- Community_Survey: mean=32.0420, std=35.1239, min=0.0000, p25=6.0000, median=19.0000, p75=47.0000, max=157.0000
- Degree Centrality_x: mean=0.0066, std=0.0035, min=0.0003, p25=0.0043, median=0.0061, p75=0.0087, max=0.0170
- Closeness Centrality_x: mean=0.0410, std=0.0383, min=0.0003, p25=0.0059, median=0.0147, p75=0.0788, max=0.1227
- Betweenness Centrality_x: mean=0.0105, std=0.0188, min=0.0000, p25=0.0000, median=0.0005, p75=0.0127, max=0.1543
- Eigenvector Centrality_x: mean=0.0184, std=0.0670, min=0.0000, p25=0.0000, median=0.0000, p75=0.0011, max=0.4668
- Community_x: mean=32.0420, std=35.1239, min=0.0000, p25=6.0000, median=19.0000, p75=47.0000, max=157.0000
- Degree Centrality_y: mean=0.0066, std=0.0035, min=0.0003, p25=0.0043, median=0.0061, p75=0.0087, max=0.0170
- Closeness Centrality_y: mean=0.0410, std=0.0383, min=0.0003, p25=0.0059, median=0.0147, p75=0.0788, max=0.1227
- Betweenness Centrality_y: mean=0.0105, std=0.0188, min=0.0000, p25=0.0000, median=0.0005, p75=0.0127, max=0.1543
- Eigenvector Centrality_y: mean=0.0184, std=0.0670, min=0.0000, p25=0.0000, median=0.0000, p75=0.0011, max=0.4668
- Community_y: mean=32.0420, std=35.1239, min=0.0000, p25=6.0000, median=19.0000, p75=47.0000, max=157.0000
- Degree Centrality: mean=0.0066, std=0.0035, min=0.0003, p25=0.0043, median=0.0061, p75=0.0087, max=0.0170
- Closeness Centrality: mean=0.0410, std=0.0383, min=0.0003, p25=0.0059, median=0.0147, p75=0.0788, max=0.1227
- Betweenness Centrality: mean=0.0105, std=0.0188, min=0.0000, p25=0.0000, median=0.0005, p75=0.0127, max=0.1543
- Eigenvector Centrality: mean=0.0184, std=0.0670, min=0.0000, p25=0.0000, median=0.0000, p75=0.0011, max=0.4668
- Community: mean=32.0420, std=35.1239, min=0.0000, p25=6.0000, median=19.0000, p75=47.0000, max=157.0000

### Top Correlations (absolute)
- Eigenvector Centrality_Survey ↔ Eigenvector Centrality: 1.0000
- Eigenvector Centrality_Survey ↔ Eigenvector Centrality_y: 1.0000
- Eigenvector Centrality_Survey ↔ Eigenvector Centrality_x: 1.0000
- Closeness Centrality_Survey ↔ Closeness Centrality_x: 1.0000
- Betweenness Centrality_Survey ↔ Betweenness Centrality: 1.0000
- Degree Centrality_x ↔ Degree Centrality: 1.0000
- Community_y ↔ Community: 1.0000
- Degree Centrality_x ↔ Degree Centrality_y: 1.0000
- Eigenvector Centrality_y ↔ Eigenvector Centrality: 1.0000
- Betweenness Centrality_x ↔ Betweenness Centrality_y: 1.0000

### Outlier Counts (IQR)
- Discussed_Religion: 324
- timeperweekactiveb: 707
- Felt_Overwhelmed: 242
- Public_Communication: 725
- Experiences_of_Depression_Score: 326
- Likelihood_Sports_Participation_Score: 9
- Active_Community_Volunteerism_Score: 141
- Professional Development_Score: 242
- Family and Personal Life_Score: 274
- Classical_Traditional_Score: 266
- Folk_Cultural_Score: 91
- Nostalgic_Easy_Score: 371
- openness: 36
- premaritalsex: 1207
- occupationmom: 1170
- happy: 396
- agreeableness: 18
- conscientiousness: 39
- neuroticism: 9
- reading_category: 96
- Degree Centrality_Survey: 126
- Betweenness Centrality_Survey: 607
- Eigenvector Centrality_Survey: 1029
- Community_Survey: 303
- Degree Centrality_x: 126
- Betweenness Centrality_x: 607
- Eigenvector Centrality_x: 1029
- Community_x: 303
- Degree Centrality_y: 126
- Betweenness Centrality_y: 607
- Eigenvector Centrality_y: 1029
- Community_y: 303
- Degree Centrality: 126
- Betweenness Centrality: 607
- Eigenvector Centrality: 1029
- Community: 303
