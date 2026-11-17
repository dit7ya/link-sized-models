// Both Node and Python read the single prefix in host.json.
const {prefix} = require('./host.json');
function checkPrefix(url) {
  if (!url.startsWith(prefix) && !url.startsWith(prefix.toUpperCase())) {
    throw Error(`Expected canonical model prefix ${prefix}`);
  }
}
module.exports = {prefix, checkPrefix};
