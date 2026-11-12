# data

Everything in `raw/` is a trimmed copy of the Himalayan Database extract that the TidyTuesday project
published as week 39 of 2020.

| file | rows | what it holds |
|------|------|---------------|
| `raw/members.csv.gz` | 76,519 | one row per person per expedition (17 of the 21 original columns kept) |
| `raw/expeditions.csv.gz` | 10,364 | one row per expedition: dates, team size, hired staff, agency, how it ended |
| `raw/peaks.csv` | 468 | peak height and first ascent year |
| `raw/tidytuesday_readme.md` | | the data dictionary that came with the files |

Source: https://github.com/rfordatascience/tidytuesday/tree/main/data/2020/2020-09-22

The Himalayan Database is compiled from the expedition archives of Elizabeth Hawley and cleaned for
TidyTuesday by Alex Cookson. It covers expeditions in the Nepal Himalaya from 1905 to spring 2019. The
pipeline keeps non-hired climbers from 1950 on, which leaves 60,521 rows after joining peak heights.

I dropped the death cause, injury type and height columns when trimming the members file because nothing
here uses them. `expedition_id` `KANG10101` appears twice in the expeditions file (1910 and 2010 rows);
the loader keeps the later one.
