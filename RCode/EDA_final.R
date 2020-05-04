library(dplyr)
library(ggplot2)
library(datetime)
library(anytime)
library(tibble)
library(tidyr)

# data  

data = read.csv("C:/Users/ranji/OneDrive/Desktop/HMDA-Loan-Prediction-with-Bias-studies-master/HMDA-Loan-Prediction-with-Bias-studies-master/NewDataCode/123.csv",skipNul = FALSE, nrows= 100000)
dim(data)
drops <- c("preapproval")
df.new = data[ , !(names(data) %in% drops)]
dim(df.new)
df.new$action_taken = factor(df.new$action_taken)
df.new = df.new %>% filter(action_taken!= c('2', '4', '5', '6', '7', '8'))



#df.new %>% mutate(df.new, applicant_ethnicity = factor(applicant_ethnicity), levels = c(1,2,3,4,5),labels = c("Hispanic/Latino","Not Hispanic/Latino", "Info not provided", "Not Available","No co-app"))


df.new= 
  
  mutate(df.new, applicant_ethnicity = factor(applicant_ethnicity, levels = c(1,2,3,4,5),labels = c("Hispanic/Latino","Not Hispanic/Latino", "Info not provided", "Not Available","No co-app"))) %>%
  
  mutate(applicant_sex = factor(applicant_sex, levels = c(1,2,3,4,5), labels = c("Male", "Female","Info not provided", "Not Available","No co-app")))%>%
  
  mutate(action_taken = factor(action_taken, levels = c(1,3), labels = c("Originated","Denied")))%>%
  
  mutate(loan_type = factor(loan_type, levels = c(1,2,3,4), labels = c("Conventional", "FHA Insured","VA Guaranteed", "FHA/RHS")))%>%
  
  mutate(property_type = factor(property_type, levels = c(1,2,3), labels = c("one-four family", "Manufactured housing","Multifamily")))%>%
  
  mutate(loan_purpose = factor(loan_purpose, levels = c(1,2,3), labels = c("Home Purchase", "Home Improvement","Refinancing")))%>%
  
 
  
  mutate(lien_status = factor(lien_status, levels = c(1,2,3,4), labels = c("First Lien", "Subordinate Lien","Not secured by lien", "Not Available")))%>%
  
  mutate(owner_occupancy = factor(owner_occupancy, levels = c(1,2,3), labels = c("Owner-occupied", "Not owner-occupied", "Not Available")))%>%
  
  mutate(hoepa_status = factor(hoepa_status, levels = c(1,2), labels = c("HOEPA loan", "Non- HOEPA loan")))%>%
  
  mutate(co_applicant_ethnicity = factor(co_applicant_ethnicity, levels = c(1,2,3,4,5),labels = c("Hispanic/Latino","Not Hispanic/Latino", "Info not provided","Not Available","No co-app")))%>%
  
  mutate(purchaser_type = factor(purchaser_type, levels = c(0,1,2,3,4,5,6,7,8,9), labels = c("Not originated", "FNMA", "GNMA", "FHLMC", "FAMC", "Pvt Securitization", "commercial/Savings Bank", "Insurance", "Affiliate","Other types")))%>%
  
  mutate(co_applicant_sex = factor(co_applicant_sex, levels = c(1,2,3,4,5), labels = c("Male", "Female","Info not provided","Not Available","No co-app" )))

View(df.new)
df.new <- na.omit(df.new, Var = c("action_taken"))
dim(df.new)


# Plotting counts for applicant_ethnicity



eth_counts <- df.new%>%
  
  group_by(action_taken, applicant_ethnicity) %>%
  
  summarise(count= n())

View(eth_counts)

ggplot(data = eth_counts, aes(x=action_taken,y= count, fill=applicant_ethnicity)) +
  
  geom_bar(position="stack", stat="identity")



# plotting counts for Different genders



gender_counts <- df.new %>%
  
  group_by(action_taken, applicant_sex) %>%
  
  summarise(count= n())

View(gender_counts)

ggplot(data = gender_counts, aes(x=action_taken,y= count, fill=applicant_sex)) +
  
  geom_bar(position="stack", stat="identity") 





#Plotting counts for different loan types



loantype_counts <- df.new %>%
  
  group_by(action_taken, loan_type) %>%
  
  summarise(count= n())

View(loantype_counts)

ggplot(data = loantype_counts, aes(x=action_taken,y= count, fill=loan_type)) +
  
  geom_bar(position="stack", stat="identity") 





#Plotting counts for HOEPA and non Hoepa Loans 



hoepa_counts <- df.new %>%
  
  group_by(action_taken, hoepa_status) %>%
  
  summarise(count= n())

View(hoepa_counts)

ggplot(data = hoepa_counts, aes(x=action_taken,y= count, fill=hoepa_status)) +
  
  geom_bar(position="stack", stat="identity") 





#Plotting counts for based on lien status





lien_counts <- df.new %>%
  
  group_by(action_taken, lien_status) %>%
  
  summarise(count= n())

View(lien_counts)

ggplot(data = lien_counts, aes(x=action_taken,y= count, fill=lien_status)) +
  
  geom_bar(position="stack", stat="identity") 



#Plotting counts for based on owner occupancy



owner_counts <- df.new %>%
  
  group_by(action_taken, owner_occupancy) %>%
  
  summarise(count= n())

View(owner_counts)

ggplot(data = owner_counts, aes(x=action_taken,y= count, fill=owner_occupancy)) +
  
  geom_bar(position="stack", stat="identity") 





#Plotting counts for based on property type 



prop_counts <- df.new %>%
  
  group_by(action_taken, property_type) %>%
  
  summarise(count= n())

View(prop_counts)

ggplot(data = prop_counts, aes(x=action_taken,y= count, fill = property_type)) +
  
  geom_bar(position="stack", stat="identity") 



loan_purpose_counts <- df.new %>%
  
  group_by(action_taken, loan_purpose) %>%
  
  summarise(count= n())

View(loan_purpose_counts)

ggplot(data = loan_purpose_counts, aes(x=action_taken,y= count, fill = loan_purpose)) +
  
  geom_bar(position="stack", stat="identity") 

purchaser_counts <- df.new %>%
  
  group_by(action_taken, purchaser_type) %>%
  
  summarise(count= n())

View(purchaser_counts)

ggplot(data = purchaser_counts, aes(x=action_taken,y= count, fill = purchaser_type)) +
  
  geom_bar(position="stack", stat="identity") 


coapp_sex_counts <- df.new %>%
  
  group_by(action_taken, co_applicant_sex) %>%
  
  summarise(count= n())

View(coapp_sex_counts)

ggplot(data = coapp_sex_counts, aes(x=action_taken,y= count, fill = co_applicant_sex)) +
  
  geom_bar(position="stack", stat="identity") 



coapp_eth_counts <- df.new %>%
  
  group_by(action_taken, co_applicant_ethnicity) %>%
  
  summarise(count= n())

View(coapp_eth_counts)

ggplot(data = coapp_eth_counts, aes(x=action_taken,y= count, fill = co_applicant_ethnicity)) +
  
  geom_bar(position="stack", stat="identity") 





############################################################################








