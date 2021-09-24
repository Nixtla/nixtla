# read the workflow template
WORKFLOW_TEMPLATE=$(cat .github/workflow-template-$1.yml)

# iterate each route in routes directory
for ROUTE in tsfeatures tsforecast; do
    echo "generating workflow for ${ROUTE}"

    # replace template route placeholder with route name
    WORKFLOW=$(echo "${WORKFLOW_TEMPLATE}" | sed "s/{{ROUTE}}/${ROUTE}/g")

    # save workflow to .github/workflows/{ROUTE}
    echo "${WORKFLOW}" > .github/workflows/${ROUTE}-$1.yml
done
