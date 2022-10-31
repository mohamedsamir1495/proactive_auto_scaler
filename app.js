const PythonShell = require('python-shell').PythonShell;

var options = {
  mode: 'text',
  pythonPath: 'D:/Program Files/Anaconda3/envs/python learn/python.exe',
  pythonOptions: ['-u'],
  scriptPath: './forecaster',
  args: ['value1', 'value2', 'value3']
};

PythonShell.run('main.py', options, function (err, results) {
  if (err) 
    throw err;
  // Results is an array consisting of messages collected during execution
  console.log('results: %j', results);
});