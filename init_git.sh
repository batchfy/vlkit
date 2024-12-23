rm -rf .git
git init
git config user.name 'Batchor'
git config user.email 'batchfy@gmail.com'
git add .
git commit -m 'init'
git remote add origin git@github.com:batchfy/vlkit
git push -u origin main -f
