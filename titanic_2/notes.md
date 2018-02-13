# Notes

## Performance

Will be evaluated as a simple accuracy. How many do I get right?

## Assumptions

- There is no simple rule that will get everything right.

## Test and training set

I created a custom training set and a custom validation set. It's not quite how A. Geron suggests but it does the job. The way he does it he manipulates numpy arrays directly. In this case I loaded the data with Pandas from the get go so I used the `DataFrame.sample()` function. This should be good enough.