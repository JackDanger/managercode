// spawn a child process and execute shell command
// taken from https://gist.github.com/millermedeiros/4724047
// author Miller Medeiros
// released under MIT License
// version: 0.1.0 (2013/02/01)

const { spawn } = require('child_process');

// execute a single shell command where "cmd" is a string
exports.exec = function(cmd, cb, stdout, stdin){
  // this would be way easier on a shell/bash script :P
  var p = spawn(cmd[0], cmd.slice(1), {stdio: 'inherit'});
  console.log(p)
  p.stdout.on('data', stdout || function() {})
  p.stdin.on('data', stdin || function() {})
  p.on('exit', function(code){
      var err = null;
      if (code) {
          err = new Error('command "'+ cmd +'" exited with wrong status code "'+ code +'"');
          err.code = code;
          err.cmd = cmd;
      }
      if (cb) cb(err);
  });
};


// execute multiple commands in series
// this could be replaced by any flow control lib
exports.series = function(cmds, cb){
    var execNext = function(){
        exports.exec(cmds.shift(), function(err){
            if (err) {
                cb(err);
            } else {
                if (cmds.length) execNext();
                else cb(null);
            }
        });
    };
    execNext();
};
