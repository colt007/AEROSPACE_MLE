# Why This Project

I took this project because I had an interest in aerospace, but my major is in computer science. So this project felt like a way to combine two things: one that I have been interested in for a long time, and another that I have been working with for years.

At first, I thought this was going to be a fairly straightforward predictive or regression problem. Predict the Remaining Useful Life of an engine, train a model, check the error, and move on.

Then I started reading the NASA documentation.

And then I started looking at the actual datasets.

That is where things became a little different.

The more I looked at the data, the less this felt like a simple regression problem with a fancy model sitting on top of it. There were different flight or operating conditions, multiple engines, many sensor readings changing over time, and most importantly, the data had a strict temporal structure that couldn't simply be shuffled around and treated like ordinary tabular data.

When I first came across articles and documentation around the NASA challenges, I honestly thought this was going to be another Titanic-style dataset.

Load the data.

Do some preprocessing.

Train a model.

Get a score.

Done.

Then I started noticing something in the way many of those experiments were being presented: a lot of models were being evaluated within a particular dataset or operating condition. That made me wonder what would happen when the conditions themselves changed. This is my observation and assumption rather than a claim about every paper or implementation, but it made me look at the problem differently.

At that point, I stopped thinking that the main challenge was going to be inventing some new hybrid architecture.

I started thinking about the data itself.

Maybe the difficult part was not going to be the model.

Maybe the difficult part was making sure the data could safely make its way into the model in the first place.

That led me to the ETL side of the project.

The dataset is fundamentally temporal. An engine is a sequence of observations over cycles. If that ordering is damaged, if windows cross from one engine into another, if the target is shifted by even one cycle, or if information from one operating condition is handled incorrectly, the model can still train perfectly well while learning from something we never intended to give it.

That realization changed how I approached the whole project.

Instead of starting with the model and trying to make the data fit it, I started from the data and worked forward.

The model would come later.

First, the data had to earn my trust.

And that is probably the part that surprised me most.

The project that I initially thought would be "train a model to predict RUL" slowly became a much bigger question:

**How do you build a machine learning system that you can actually trust before you ever look at the final prediction?**

That is where this project really started.
