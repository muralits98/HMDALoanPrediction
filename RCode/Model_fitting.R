library(grid)
library(gridExtra)
library(nnet)

#######Preparing the data#######

data = read.csv("C:/Users/ranji/OneDrive/Desktop/HMDA-Loan-Prediction-with-Bias-studies-master/HMDA-Loan-Prediction-with-Bias-studies-master/NewDataCode/123.csv",skipNul = FALSE, nrows= 100000)

dim(data)
drops <- c("preapproval")
df.new = data[ , !(names(data) %in% drops)]
dim(df.new)
df.new$action_taken = factor(df.new$action_taken)
df.new = df.new %>% filter(action_taken!= c('2', '4', '5', '6', '7', '8'))

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



# prepare training and test data 



set.seed(100)
Train <- createDataPartition(df.new$action_taken, p=0.7, list=FALSE)
training <- data.frame(df.new[ Train, ])
testing <- data.frame(df.new[ -Train, ])




######## fitting model using polynomial function#######

library(boot)
set.seed(999)
cv.error = rep(NA,5)
for(i in 1:5)
{
  fit1 = glm(action_taken ~ poly(number_of_1_to_4_family_units,i) + poly(number_of_owner_occupied_units,i)+ poly(loan_amount_000s,i)+poly(tract_to_msamd_income,i) + poly(hud_median_family_income,i) + poly(minority_population,i) + poly(population,i) + poly(applicant_income_000s,i) + lien_status +loan_type+ property_type +loan_purpose +owner_occupancy + purchaser_type + applicant_ethnicity +co_applicant_ethnicity + applicant_sex +co_applicant_sex  + hoepa_status, data= training, family = binomial)
  cv.error[i+1] = cv.glm(training, fit1, K = 4)$delta[1]
  print(i)
}

fit1 = glm(action_taken ~ 1, data = training, family = "binomial")
cv.error[1] = cv.glm(training , fit1, K =4)$delta[1]

cv.error
plot(cv.error, type = "b")


# identified i to be 2 as it gives the lowest cross validation error- use this in testing data 

####fitting model using the polynomial function with i =2#############

library(caret)
library(e1071)
fit2 = glm(action_taken ~ poly(number_of_1_to_4_family_units,2) + poly(number_of_owner_occupied_units,2)+ poly(loan_amount_000s,2)+poly(tract_to_msamd_income,2) + poly(hud_median_family_income,2) + poly(minority_population,2) + poly(population,2) + poly(applicant_income_000s,2) + lien_status +loan_type+ property_type +loan_purpose +owner_occupancy + purchaser_type + applicant_ethnicity +co_applicant_ethnicity + applicant_sex +co_applicant_sex  + hoepa_status, data= training,  family = "binomial")
#exp(coef(fit2$finalModel))
#head(predict(fit2, type = "response"))

#drop.test <- c("action_taken")
#testing = testing[ , !(names(testing) %in% drop.test)]

pred = predict(fit2, testing, type = "response")
pred.logit <- rep('Originated',length(pred))
pred.logit[pred>=0.5] <- 'Denied'

confusionMatrix(as.factor(pred.logit), testing$action_taken)


########fitting using natural splines #########
library(splines)
library(gam)
library(boot)
set.seed(999)
cv.error = rep(NA,5)
for(i in 1:5)
{
  fit2 = glm(action_taken ~ ns(number_of_1_to_4_family_units,i) +s(number_of_owner_occupied_units,i)+ ns(loan_amount_000s,i)+ns(tract_to_msamd_income,i) + ns(hud_median_family_income,i) + ns(minority_population,i) + ns(population,i) + ns(applicant_income_000s,i) + lien_status +loan_type+ property_type +loan_purpose +owner_occupancy + purchaser_type + applicant_ethnicity +co_applicant_ethnicity + applicant_sex +co_applicant_sex  + hoepa_status, data= training, family = binomial)
  cv.error[i+1] = cv.glm(training, fit2,  K = 4)$delta[1]
  print(i)
}

fit2 = glm(action_taken ~ 1, data = training, family = "binomial")
cv.error[1] = cv.glm(training , fit2, K =4)$delta[1]

cv.error

plot(cv.error, type = "b")


library(caret)
library(e1071)
fit3 = glm(action_taken ~ ns(number_of_1_to_4_family_units,4) + ns(number_of_owner_occupied_units,4)+ ns(loan_amount_000s,4)+ns(tract_to_msamd_income,4) + ns(hud_median_family_income,4) + ns(minority_population,4) + ns(population,4) + ns(applicant_income_000s,4) + lien_status +loan_type+ property_type +loan_purpose +owner_occupancy + purchaser_type + applicant_ethnicity +co_applicant_ethnicity + applicant_sex +co_applicant_sex  + hoepa_status, data= training,  family = "binomial")
#exp(coef(fit2$finalModel))
#head(predict(fit2, type = "response"))



pred = predict(fit3, testing, type = "response")
pred.logit <- rep('Originated',length(pred))
pred.logit[pred>=0.5] <- 'Denied'

confusionMatrix(as.factor(pred.logit), testing$action_taken)
AIC(fit3)



fit4 = glm(action_taken ~ ns(number_of_1_to_4_family_units,2) + ns(number_of_owner_occupied_units,2)+ ns(loan_amount_000s,2)+ns(tract_to_msamd_income,2) + ns(hud_median_family_income,2) + ns(minority_population,2) + ns(population,2) + ns(applicant_income_000s,2) + lien_status +loan_type+ property_type +loan_purpose +owner_occupancy + purchaser_type + applicant_ethnicity +co_applicant_ethnicity + applicant_sex +co_applicant_sex  + hoepa_status, data= training,  family = "binomial")

pred = predict(fit3, testing, type = "response")
pred.logit <- rep('Originated',length(pred))
pred.logit[pred>=0.5] <- 'Denied'

confusionMatrix(as.factor(pred.logit), testing$action_taken)
AIC(fit4)
