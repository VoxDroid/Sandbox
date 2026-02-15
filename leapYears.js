const leapYears = function(year) {
  if (year % 4 === 0) {
    if (year % 100 === 0 && year % 400 !== 0) {
      return false;
    }
    return true;
  }
  return false;
};

// When comparing my solution to the actual solution. The actual solution
// declare a variable to make the code more readable. My solution has nested if
// while the actual solution used logical operators more to avoide nested if.


// Do not edit below this line
module.exports = leapYears;
