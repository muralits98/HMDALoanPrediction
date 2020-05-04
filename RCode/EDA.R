library(dplyr)
library(ggplot2)
library(datetime)
library(anytime)
# data below has 
data = read.csv("C:/Users/ranji/OneDrive/Desktop/HMDA-Loan-Prediction-with-Bias-studies-master/HMDA-Loan-Prediction-with-Bias-studies-master/NewDataCode/123.csv",skipNul = FALSE, nrows= 100000)

View(data)
head(data)
dim(data)
colnames(data)

data1 = data%>%
  filter(owner_occupancy != c('3') & 
           preapproval != c('3') & lien_status != c('4') &
           action_taken != c('7', '8'))#& applicant_ethnicity!= c('3', '4', '5') & applicant_sex!= c('3','4', '5') & lien_status!= c('4') & co_applicant_sex!= c('3','4', '5') & co_applicant_ethnicity != c('4', '5','3'))


# data1 = data%>%
#   #filter(owner_occupancy != c('3') & 
#            #preapproval != c('3') & lien_status != c('4') &
#            filter(action_taken != c('7', '8'))


View(data1)
dim(data1)
#df = na.omit(df)
#dim(df)
#install.packages(c("tibble", "dplyr", "tidyr"))
xtabs(~ action_taken + applicant_sex, data=df)

df= 
  mutate(data1, applicant_ethnicity = factor(applicant_ethnicity, levels = c(1,2,3,4,5),labels = c("Hispanic/Latino","Not Hispanic/Latino", "Info not provided", "NA","No co-app"))) %>%
  mutate(applicant_sex = factor(applicant_sex, levels = c(1,2,3,4,5), labels = c("Male", "Female","Info not provided", "NA","No co-app")))%>%
  mutate(action_taken = factor(action_taken, levels = c(1,2,3,4,5,6), labels = c("Originated", "approved","denied", "withdrawn", "incomplete", "purchased")))%>%
  mutate(loan_type = factor(loan_type, levels = c(1,2,3,4), labels = c("Conventional", "FHA Insured","VA Guaranteed", "FHA/RHS")))%>%
  mutate(property_type = factor(property_type, levels = c(1,2,3), labels = c("one-four family", "Manufactured housing","Multifamily")))%>%
  mutate(loan_purpose = factor(loan_purpose, levels = c(1,2,3), labels = c("Home Purchase", "Home Improvement","Refinancing")))%>%
  mutate(preapproval = factor(preapproval, levels = c(1,2), labels = c("Requested", "Not Requested")))%>%
  mutate(lien_status = factor(lien_status, levels = c(1,2,3), labels = c("First Lien", "Subordinate Lien","Not secured by lien")))%>%
  mutate(owner_occupancy = factor(owner_occupancy, levels = c(1,2), labels = c("Owner-occupied", "Not owner-occupied")))%>%
  mutate(hoepa_status = factor(hoepa_status, levels = c(1,2), labels = c("HOEPA loan", "Non- HOEPA loan")))%>%
  mutate(co_applicant_ethnicity = factor(co_applicant_ethnicity, levels = c(1,2,3,4,5),labels = c("Hispanic/Latino","Not Hispanic/Latino", "Info not provided","NA","No co-app")))%>%
  mutate(purchaser_type = factor(purchaser_type, levels = c(0,1,2,3,4,5,6,7,8,9), labels = c("Not originated", "FNMA", "GNMA", "FHLMC", "FAMC", "Pvt Securitization", "commercial/Savings Bank", "Insurance", "Affiliate","Other types")))%>%
  mutate(co_applicant_sex = factor(co_applicant_sex, levels = c(1,2,3,4,5), labels = c("Male", "Female","Info not provided","NA","No co-app" )))


df$loan_type = factor(data1$loan_type, levels = c(1,2,3,4), labels = c("Conventional", "FHA Insured","VA Guaranteed", "FHA/RHS"))
View(df)
dim(df)
df1 = na.omit(df)
dim(df1)


# df = data[data$action_taken != "preapproval denied",]
# levels(df$action_taken)
# df$action_taken <- factor(df$action_taken)
# levels(df$action_taken)
# 
# df2 = df[df$action_taken != "preapproval approved",]
# levels(df$action_taken)
# df2$action_taken <- factor(df2$action_taken)
# levels(df2$action_taken)
# View(df2)

#Plotting Action_taken for each ethnicity 
#'%notin%' <- Negate(`%in%`)

#df3<- df2[df2$applicant_ethnicity %notin% c("NA", "No info Provided", "No co-app"),]
#df3<- df2[df2$applicant_ethnicity== c("NA", "No info Provided", "No co-app"),]
#filter(df2, applicant_ethnicity!= "NA" & applicant_ethnicity!= "No info Provided")

# Plotting counts for applicant_ethnicity

eth_counts <- df1%>%
  group_by(action_taken, applicant_ethnicity) %>%
  summarise(count= n())
View(eth_counts)
ggplot(data = eth_counts, aes(x=action_taken,y= count, fill=applicant_ethnicity)) +
  geom_bar(position="stack", stat="identity")

# plotting counts for Different genders

gender_counts <- df1 %>%
  group_by(action_taken, applicant_sex) %>%
  summarise(count= n())
View(gender_counts)
ggplot(data = gender_counts, aes(x=action_taken,y= count, fill=applicant_sex)) +
  geom_bar(position="stack", stat="identity") 


#Plotting counts for different loan types

loantype_counts <- df1 %>%
  group_by(action_taken, loan_type) %>%
  summarise(count= n())
View(loantype_counts)
ggplot(data = loantype_counts, aes(x=action_taken,y= count, fill=loan_type)) +
  geom_bar(position="stack", stat="identity") 


#Plotting counts for HOEPA and non Hoepa Loans 

hoepa_counts <- df1 %>%
  group_by(action_taken, hoepa_status) %>%
  summarise(count= n())
View(hoepa_counts)
ggplot(data = hoepa_counts, aes(x=action_taken,y= count, fill=hoepa_status)) +
  geom_bar(position="stack", stat="identity") 
  

#Plotting counts for based on lien status


lien_counts <- df1 %>%
  group_by(action_taken, lien_status) %>%
  summarise(count= n())
View(lien_counts)
ggplot(data = lien_counts, aes(x=action_taken,y= count, fill=lien_status)) +
  geom_bar(position="stack", stat="identity") 

#Plotting counts for based on owner occupancy

owner_counts <- df1 %>%
  group_by(action_taken, owner_occupancy) %>%
  summarise(count= n())
View(owner_counts)
ggplot(data = owner_counts, aes(x=action_taken,y= count, fill=owner_occupancy)) +
  geom_bar(position="stack", stat="identity") 


#Plotting counts for based on property type 

prop_counts <- df1 %>%
  group_by(action_taken, property_type) %>%
  summarise(count= n())
View(prop_counts)
ggplot(data = prop_counts, aes(x=action_taken,y= count, fill = property_type)) +
  geom_bar(position="stack", stat="identity") 

############################################################################

xtabs(~ applicant_sex + action_taken, data=df1)

summary(df$applicant_sex)

nrow(filter(df, action_taken == "approved" & applicant_sex == "Male"))
nrow(filter(df, action_taken == "approved" & applicant_sex == "Female"))

nrow(filter(df, action_taken == "denied" & applicant_sex == "Male"))
nrow(filter(df, action_taken == "denied" & applicant_sex == "Female"))


library(grid)
library(gridExtra)
library(nnet)

# prepare training and test data 

set.seed(100)
trainingRows <- sample(1:nrow(df1), 0.7*nrow(df1))
training <- df1[trainingRows, ]
test <- df1[-trainingRows, ]
# df <- mutate(df, loan_type = factor(loan_type)) %>%
#   mutate(loan_purpose = as.factor(loan_purpose)) %>%
#   within(loan_purpose <- relevel(loan_purpose, ref = 1)) %>%
#   mutate(applicant_ethnicity = factor(applicant_ethnicity)) %>%
#   within(applicant_ethnicity <- relevel(applicant_ethnicity, ref = 3)) %>%
#   mutate(property_type = factor(property_type)) %>%
#   within(property_type <- relevel(property_type, ref = 2)) %>%
#   mutate(co_appplicant_sex = factor(co_applicant_sex))
model <- multinom(action_taken ~ number_of_1_to_4_family_units + number_of_owner_occupied_units + tract_to_msamd_income +
               + hud_median_family_income + minority_population + population + lien_status +loan_type+ property_type               
             +loan_purpose +owner_occupancy +loan_amount_000s +preapproval+                  
             state_code + applicant_ethnicity +co_applicant_ethnicity +     
             applicant_sex +co_applicant_sex  +           
             applicant_income_000s + hoepa_status, data=training)
summary(model)

#training accuracy

training$predicted <- predict(model, newdata = training, "class")
ctable <- table(training$action_taken, training$predicted)
round((sum(diag(ctable))/sum(ctable))*100,2)

#[1] 62.09


# testing accuracy


test$predicted <- predict(model, newdata = test, "class")

# Building classification table
ctable <- table(test$action_taken, test$predicted)

# Calculating accuracy - sum of diagonal elements divided by total obs

#[1] 61.73

library(survival)
library(ggfortify)




