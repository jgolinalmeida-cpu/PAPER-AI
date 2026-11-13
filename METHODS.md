# Methods guide

## 1. The idea in one paragraph

The two papers you sent describe the same economy from two sides. Acemoglu (2024) says the aggregate gain from AI equals the share of tasks AI affects times the cost saving on those tasks, and that this number is small. The NANDA report (2025) says the gain is also very unequal: a few firms that integrate AI into workflows capture value, most pilots return nothing, and a large amount of AI use happens through personal tools the firm never approved. So the model needs three things at once: a common AI shock, a response that differs across firms, and a separate variable for shadow use. On top of that sits the financial question of whether stock prices have run ahead of those modest fundamentals.

## 2. Which puremacro models I chose, and why

I received one link (the puremacro quickstart). You mentioned two sources of models, so the second link may not have come through. Send it and I will compare.

| Question | Model | puremacro function | Why this one |
|---|---|---|---|
| A. What does an AI shock do to the economy? | Local projections | `lp.lp_hac` | Few assumptions, works with a short sample, easy to explain. A Cholesky VAR (`var.identify.cholesky_svar`) is the cross-check. |
| B1. Do AI-exposed and shadow-heavy firms respond differently to the same shock? | Panel local projections with interactions | `lp.panel_lp`, plus a two-step version in `panel.py` | This is the direct test of "some firms benefit, others do not". |
| B2. Does adopting AI raise profits? | Staggered difference-in-differences | `did.callaway_santanna`, `lp.lp_did` | Firms adopt at different dates. Older two-way fixed effects estimators are biased in that case. |
| C. What does shadow AI do? | Two-way fixed effects regression | written in `panel.py` | Simple and transparent. Uses the shadow gap variable defined below. |
| D. Is there a bubble? What did the NVIDIA case do? | Phillips-Shi-Yu test and an event study | written in `bubble.py` | puremacro has no explosive-root test, so I wrote one. |
| E. What does it add up to? | Acemoglu's formula with a shadow margin | written in `calibrate.py` | Links your estimates back to the paper. |

Models in puremacro I looked at and left out: HANK and DSGE models (they need a full calibrated economy and have no AI adoption margin, and four years of data cannot discipline them), FAVAR and GVAR (need far more data), nowcasting, sovereign default and climate modules (different questions).

## 3. The equations

**A. Aggregate shock.** The AI shock is the part of the AI stock basket's monthly return that the market return, its own lags and lagged macro conditions do not explain. It is scaled to one standard deviation. For each horizon h:

    y[t+h] - y[t-1] = a_h + b_h * shock[t] + lags + error

The bust scenario multiplies b_h by the size of a large fall (default 30 percent relative to the market). This is a linear extrapolation and should be described that way.

**B1. Different responses.** Step one, for each quarter: regress each unit's change in profit margin on its AI exposure and its shadow intensity. Step two: regress those slopes on the AI shock. The result says how much more a highly exposed unit gains from a positive AI shock than a low-exposure unit.

**B2. Adoption effect.** ATT(g, t) is the average change in profit margin for units that integrated AI in quarter g, measured at quarter t, relative to units that never adopted. The code estimates it twice: once for integrated AI and once for pilots that never integrated. The NANDA claim predicts a positive effect for the first group and zero for the second.

**C. Shadow AI.**

    shadow_ai = max(worker_ai - official_ai, 0)
    y[i,t] = unit effect + time effect + b1*official_ai + b2*shadow_ai + b3*(official_ai x shadow_ai) + exposure x time effects + error

worker_ai is the share of workers who use AI at work. official_ai is the share covered by an employer-approved tool. The gap is a lower bound on shadow use. Run it for productivity and for profit margin. The NANDA report claims personal tools raise individual productivity but not the firm's P&L, which would show up as b2 above zero for productivity and near zero for margin.

**D. Bubble.** The Phillips-Shi-Yu test asks whether a price series behaves like an explosive process over some window. The BSADF statistic is computed for every date and compared with simulated critical values. Dates above the line for long enough are flagged. Apply it to the AI basket relative to the market, and if you can, to a price-to-earnings or price-to-sales ratio, which is the cleaner version. The event study estimates a market model before each event and reports the abnormal return of NVIDIA, the AI basket and a non-AI basket. If the non-AI basket also falls, the case is spreading beyond AI.

**E. Calibration.** Acemoglu's equation: TFP gain = share of tasks affected x average cost saving = (0.199 x 0.23) x (0.27 x 0.535) = 0.66 percent over ten years. The extension adds shadow tasks with a pass-through rate between zero and one. The pass-through is an assumption until you estimate b2 in module C.

## 4. Data you need

| Variable | Suggested public source | Notes |
|---|---|---|
| Stock prices | Yahoo Finance through `yfinance` | Loader included. |
| Macro series | FRED | Loader included. Check the series IDs in `config.py`. |
| official_ai | Census Business Trends and Outlook Survey (BTOS), AI use by sector | The question wording changed in November 2025 from use "in producing goods or services" to use "in any business function". Treat that as a break. |
| worker_ai | Bick, Blandin and Deming's Real-Time Population Survey; the BTOS AI supplement on worker-task use (Census working paper CES-WP-26-25) | Worker surveys are less frequent. Interpolate between waves and say so. |
| exposure | Eloundou et al. (2024) occupation exposure, averaged to industry | The same measure Acemoglu uses. |
| profit_margin | Census Quarterly Financial Report by industry, or Compustat if the university has WRDS | |
| log_productivity | BLS labor productivity by industry | |
| adopt_date, pilot_date | Only observable at firm level (10-K text, earnings calls, your own survey) | With sector data, skip B2 or use the date the sector's official use crosses a threshold. |

A sector panel (about 18 sectors by 14 quarters) is realistic with free data. A firm panel is better but needs Compustat plus a text measure of adoption.

## 5. Limits you should state in the paper

1. The sample is short. ChatGPT launched in November 2022, so there are about 16 quarters. With a common shock, the number of quarters is the real sample size, however many firms you have.
2. In simulations with 31 quarters, the 90 percent bands of the two-step estimator covered the truth 80 to 90 percent of the time. puremacro's panel LP with clustered or Driscoll-Kraay errors covered it only 50 to 75 percent of the time. That is why the two-step version is the default and the panel LP is the cross-check.
3. Local projections are biased toward zero in short samples at longer horizons (Herbst and Johannsen 2024). In the simulations the response after two quarters was about 0.20 against a true 0.29.
4. Shadow use and AI exposure are highly correlated across sectors. The data may not be able to tell them apart. Report both and do not over-read the shadow coefficient.
5. Adoption is a choice. Firms that integrate AI may differ in ways the pre-trend test does not catch. `puremacro.did.honest_did` gives a sensitivity analysis.
6. The NANDA figures come from 52 interviews and 153 survey responses and the authors call them directional. Use them to motivate hypotheses, not as data points.
7. A bubble test detects explosive prices. It cannot prove prices are unjustified. An event study on a case that is still unfolding will change as new dates arrive.

## 6. What I would add

- **J-curve check.** Brynjolfsson, Rock and Syverson (2021) argue that new general purpose technologies lower measured productivity at first because firms spend on unmeasured intangibles. The adoption event study already shows this if it exists: look for a dip in the first quarters after adoption.
- **State dependence.** `puremacro.lp.lp_state_dep` can test whether the economy responds differently to AI shocks when the bubble statistic is above its critical value.
- **Better shock identification.** Use NVIDIA's abnormal returns on earnings days as an instrument in `puremacro.lp.lp_iv`.
- **Downside risk.** `puremacro.lp.lp_quantile` estimates the effect of an AI shock on the lower tail of growth, which matters more for a bust than the average effect.
- **Investment channel.** AI capital spending is now a visible share of GDP growth. Add information-processing investment (in `config.py`) as an outcome in module A and treat it as the main transmission channel of a bust.

## 7. Sources

**Papers you supplied**
- Acemoglu, D. (2024). The Simple Macroeconomics of AI. NBER Working Paper 32487. http://www.nber.org/papers/w32487
- Challapally, A., Pease, C., Raskar, R. and Chari, P. (2025). The GenAI Divide: State of AI in Business 2025. MIT NANDA.

**Software**
- puremacro documentation. https://jalonso1979.github.io/puremacro/docs/quickstart/ (version 6.4.0 used)

**Methods**
- Jordà, Ò. (2005). Estimation and Inference of Impulse Responses by Local Projections. American Economic Review 95(1).
- Callaway, B. and Sant'Anna, P. (2021). Difference-in-Differences with Multiple Time Periods. Journal of Econometrics 225(2).
- Dube, A., Girardi, D., Jordà, Ò. and Taylor, A. (2023). A Local Projections Approach to Difference-in-Differences. NBER Working Paper 31184.
- Driscoll, J. and Kraay, A. (1998). Consistent Covariance Matrix Estimation with Spatially Dependent Panel Data. Review of Economics and Statistics 80(4).
- Herbst, E. and Johannsen, B. (2024). Bias in Local Projections. Journal of Econometrics 240(1).
- Phillips, P., Shi, S. and Yu, J. (2015). Testing for Multiple Bubbles: Historical Episodes of Exuberance and Collapse in the S&P 500. International Economic Review 56(4), 1043-1078. https://doi.org/10.1111/iere.12132
- MacKinlay, A. C. (1997). Event Studies in Economics and Finance. Journal of Economic Literature 35(1).
- Hulten, C. (1978). Growth Accounting with Intermediate Inputs. Review of Economic Studies 45(3).
- Brynjolfsson, E., Rock, D. and Syverson, C. (2021). The Productivity J-Curve. American Economic Journal: Macroeconomics 13(1), 333-372. https://www.aeaweb.org/doi/10.1257/mac.20180386

**AI adoption evidence and data**
- Bick, A., Blandin, A. and Deming, D. (2024). The Rapid Adoption of Generative AI. NBER Working Paper 32966. https://www.nber.org/system/files/working_papers/w32966/w32966.pdf
- Humlum, A. and Vestergaard, E. (2025). Large Language Models, Small Labor Market Effects. BFI Working Paper 2025-56. https://bfi.uchicago.edu/wp-content/uploads/2025/04/BFI_WP_2025-56-3.pdf
- Bonney, K. et al. (2024). Tracking Firm Use of AI in Real Time. Census CES-WP-24-16. https://www.census.gov/hfp/btos/downloads/CES-WP-24-16.pdf
- U.S. Census Bureau (2026). Large Firms With at Least 20 Employees Biggest AI Users. https://census.gov/library/stories/2026/05/ai-use-businesses.html
- U.S. Census Bureau (2026). The Microstructure of AI Diffusion. CES-WP-26-25. https://www.census.gov/library/working-papers/2026/adrm/CES-WP-26-25.html
- Eloundou, T., Manning, S., Mishkin, P. and Rock, D. (2024). GPTs are GPTs. Science 384.

**AI investment and the NVIDIA case (news, November 2026)**
- Goldman Sachs Research. Global AI investment is forecast to exceed $1 trillion in 2026. https://www.goldmansachs.com/insights/articles/global-investment-is-forecast-to-exceed-1-trillion-in-2026
- Bloomberg (Nov 8, 2026). Nvidia Board to Probe Spying Case After CEO Jensen Huang Admits Infidelity. https://www.bloomberg.com/news/articles/2026-11-08/nvidia-board-to-probe-spying-case-after-ceo-jensen-huang-admits-infidelity
- Bloomberg (Nov 12, 2026). Jensen Huang Removed as Nvidia CEO, Board Appoints Kress. https://www.bloomberg.com/news/articles/2026-11-12/jensen-huang-face-of-the-ai-boom-ousted-by-nvidia-board-as-ceo
- Axios (Nov 12, 2026). Jensen Huang resigns as Nvidia CEO. https://www.axios.com/2026/11/12/nvidia-jensen-huang-out
- Associated Press via Anchorage Daily News (Nov 10, 2026). Nvidia and Jensen Huang face shareholder lawsuit. https://www.adn.com/nation-world/2026/11/10/nvidia-and-jensen-huang-face-shareholder-lawsuit-alleging-investors-were-misled-about-spy-affair/
- TechCrunch (Nov 9, 2026). The race is on to sue Nvidia. https://techcrunch.com/2026/11/09/the-race-is-on-to-sue-nvidia/

References without a link are standard citations I gave from memory. Check volume and page numbers before you put them in a bibliography.
