#article #modelling #data

# Summary

Year: 2015

The article shows results of (continuous) income prediction on 5 real datasets of Turkish banks.

* the datasets have 10k-13k observations
* they were split 70:30
* 16 models were evaluated
  * 1-stage
    * linear (regressions)
    * non-linear (e. g. regression trees)
  * 2-stage
    * OLS at 1st stage to predict, then a non-linear method to predict residuals
    * then the residuals are added to the original estimate
* Interesting metrics models were evaluated on:
  * RMSE
  * MAE
  * R2
  * Pearson correlation
  * Spearman correlation
  * Hit rate
    * 1 if a prediction is within a certain percentage of the real value (here 15%)
    * 0 otherwise
  * AUC curve
    * based on how well the model distinguishes if an income is below or above average
  * Preciseness
    * do the predictions fall into a certain acceptable range? (i. e. are the predictions grossly over of underestimated?)
* authors note how R2 is not really a good or only method to consider when evaluating a model, but should be coupled with another method (e.g hit rate)
* models were rank ordered based on performance in the following way
  * determined the relative rank of each model on all 5 datasets
  * took the average of the ranks using Friedman's test and calculated an overall rank
    * nonparametric test with H0: all models perform the same, based on their ranking for a performance metric
  * Hommel's test to find the pairwise comparisons which produce rank differences with 95% confidence level
  * the differences between the models were small

## Findings

- two-stage models perform the best
- simple linear regression still works pretty well
- the worst performance was with models where the target was transformed in order to comply with normality assumptions in models!
- while some absolute metrics like R2 were low, it does not imply a bad model (actually the paper doesn't report the values, only a rank ordering of performances)

### Interesting

- Turkish banks instituted a "single limit" policy for credit card limits, where a limit cannot be more than 4x the customers income

# Futher

Potential further analysis could be made on 

- different segments like retired, self-employed, etc.
- predict nominal levels of income instead of real value
- compare different countries

# COO - ideas

1. As a 2nd part, predict residuals and add them to the 1st prediction

2. Split predictions into groups, produce most common residuals in each and adjust each group with that - basically same as current approach, but will potentially overestimate some and underestimate some even more

3. Run whole part 2 and Predict final as weighted sum of relative probabilities and incomes in the group - not really more useful than residuals, maybe min(estimated, weighted_sums)? - not ok, just overestimates more


# Implemented 

* hit rate and preciseness that takes data from the modelling node
  * percentages and percentiles can be modified
* comparison to "worst case" scenario when 50% are overestimated 100+% and 50% are underestimated by -100+%

```SAS
libname lib "...";
%let target = income_real;
%let prediction = income_estimate;
%let input = table1;
%let outs = stats;
%let outd = data1;
%let outm = &lib.metrics;

data &outd;
    set &input;
    pct_diff = ((&prediction/&target)-1)*100;
run;

proc means data = &input min max p1 p5 p10 p90 p95 p99;
	var &target;
	output out = &outs p1 = p1 p5 = p5 p10 = p10 p90=p90 p95 = p95 p99 = p99    min=mn max = mx;
run;

proc sql;
	select _FREQ_, p1, p99, p5, p95, p10, p90, mn, mx into :N, :p1, :p99, :p5, :p95, :p10, :p90, :mn, :mx
	from &outs;
quit;

%let hit_pct = 50;
%let left_border=&p5;
%let right_border=&p95;
/** worst-case is scenario where all of the data is either over 100% over or underestimated**/
/**ratio close to 1 is bad, close to 0 is good**/
%let worst_case=1;
/***define hit-rate and preciseness****/
proc sql;
	create table &outm as
	select sum(hit)/&N as hit_rate format percent8.1, sum(precise)/&N as preciseness format percent8.1, 
			(sum(worst)/&N)/&worst_case as worst_compared
	from (
		select case
					when abs(pct_diff)<&hit_pct then 1
					else 0
				end as hit,
				case
					when income_estimate between &left_border and &right_border then 1
					else 0
				end as precise,
                case
                    when abs(pct_diff)>=100 then 1
                    else 0
                end as worst
		from &input
	);
quit;
```

# Reference

Kibekbaev, Azamat, and Ekrem Duman. “Benchmarking Regression Algorithms for Income Prediction Modeling.” *Information Systems* 61 (October 2016): 40–52. https://doi.org/10.1016/j.is.2016.05.001.

