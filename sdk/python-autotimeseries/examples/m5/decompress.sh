for file in `find . -name '*output.tar.gz'`; do \
  DRN_NAME=$(dirname $file)
  sudo tar -xzvf "${file}" -C $DRN_NAME ;
done
