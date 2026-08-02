pipeline {
    agent any

    options {
        disableConcurrentBuilds()
    }

    environment {
        IMAGE_NAME = "nwachukwuchukwuka123/wave-racer"
        GIT_USER   = "santos-alaska"
        GIT_EMAIL  = "santosalaska123@gmail.com"
    }

    stages {

        stage('Checkout') {
            steps {
                checkout scm
            }
        }

        stage('Build and Push Image') {
            when {
                branch 'main'
            }

            steps {
                script {
                    env.IMAGE_TAG = "build-${BUILD_NUMBER}"

                    withCredentials([usernamePassword(
                        credentialsId: 'docker-creds',
                        usernameVariable: 'DOCKER_USER',
                        passwordVariable: 'DOCKER_PASS'
                    )]) {
                        sh """
                        docker build -t ${IMAGE_NAME}:${IMAGE_TAG} .

                        echo "\$DOCKER_PASS" | \
                        docker login -u "\$DOCKER_USER" --password-stdin

                        docker push ${IMAGE_NAME}:${IMAGE_TAG}
                        """
                    }
                }
            }
        }

        stage('Run in Docker') {
            when {
                branch 'main'
            }

            steps {
                sh """
                # Stop the old container if it is running
                docker stop wave-racer-container || true

                # Remove the old container if it exists
                docker rm wave-racer-container || true

                # Pull the newly pushed image
                docker pull ${IMAGE_NAME}:${IMAGE_TAG}

                # Run the new container
                docker run -d \
                    --name wave-racer-container \
                    -p 8080:80 \
                    --restart unless-stopped \
                    ${IMAGE_NAME}:${IMAGE_TAG}
                """
            }
        }

        stage('Update K8s Manifest') {
            when {
                branch 'main'
            }

            steps {
                script {
                    withCredentials([usernamePassword(
                        credentialsId: 'github-creds',
                        usernameVariable: 'GIT_USERNAME',
                        passwordVariable: 'GIT_TOKEN'
                    )]) {
                        sh """
                        set -e

                        git config user.name "$GIT_USER"
                        git config user.email "$GIT_EMAIL"

                        git fetch origin
                        git checkout main
                        git reset --hard origin/main

                        sed -i \
                        "s|image:.*|image: ${IMAGE_NAME}:${IMAGE_TAG}|" \
                        k8s/deployment.yml

                        git add k8s/deployment.yml

                        git diff --cached --quiet || \
                        git commit -m "Updated image to ${IMAGE_TAG}"

                        git push \
                        https://\$GIT_USERNAME:\$GIT_TOKEN@github.com/santos-alaska/wave-racer.git \
                        main
                        """
                    }
                }
            }
        }
    }
}
