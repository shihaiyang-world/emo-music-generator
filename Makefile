build-prod:
	$(eval tag := $(shell cat version))
	docker build -t docker.shuiditech.com/devops/vllm-openai:$(tag) .
	docker push docker.shuiditech.com/devops/vllm-openai:$(tag)

release:
	$(eval tag := $(shell cat version))
	git tag -a $(tag) -m "Branch: `git branch | grep \* | awk '{print $$2}'`"
	git push origin refs/tags/$(tag)
	$(shell echo $$(awk -F "." '{print $$1"."$$2"."($$3+1)}' version) > version)
	git commit -am "prepare for next development iteration"
	git push origin HEAD
