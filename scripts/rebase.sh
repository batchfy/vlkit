set -ex

rm -rf .git
git init && git add . && git commit -m 'rebase'
git branch -m master master
git checkout master
git remote add origin git@github.com:vlkit/vlkit
git push -u origin master -f
echo done
