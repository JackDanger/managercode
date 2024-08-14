# Manager Code

### Your team builds your product, you build your team.

##### Visualize committer connectedness across all repos
```
./calculate-shared-git-contributions.py --directory /path/to/repos \
                                        [--since '90 days ago'] \
                                        [--ignore github noreply root ec2-user] \
                                        [--publish s3://bucket/public/path]
```

