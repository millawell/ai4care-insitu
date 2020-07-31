# Required R packages for the case study
packages <- c("dplyr",
              "ggplot2",
              "tidyverse",
              "haven",
              "httr",
              "knitr",
              "lfe",
              "lme4",
              "miceadds",
              "scales",
              "stats",
              "readr",
              "rmarkdown")
ipak(packages) #Install or activates required packages

#setwd("C:/Users/User/Desktop")

raw <- read.table(file = 'train.tsv', sep = '\t', header = TRUE)
head(raw)

df <- raw
df$Mitarbeiter.ID <- gsub( "[[]", "", paste(df$Mitarbeiter.ID))
df$Mitarbeiter.ID <- gsub( "[]]", "", paste(df$Mitarbeiter.ID))
df$Mitarbeiter.ID <- as.character(df$Mitarbeiter.ID)
df$Mitarbeiter.ID <- gsub( " ", ",", paste(df$Mitarbeiter.ID))

na.strings=c(""," ","NA")

df <- df %>% separate_rows(Mitarbeiter.ID)
write.table(df, file='train_cleaned.tsv', quote=FALSE, sep='\t', col.names = NA)
df <- read.table(file = 'train_cleaned.tsv', sep = '\t', header = TRUE, na.strings=c("","NA"))
df <- df %>% filter(!is.na(Mitarbeiter.ID))
write.table(df, file='train_cleaned.tsv', quote=FALSE, sep='\t', col.names = NA)
head(df)