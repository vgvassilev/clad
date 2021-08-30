---
title: "Open Projects"
layout: textlay
excerpt: "Projects"
sitemap: false
permalink: /open_projects
---

# Open Projects

<nav>
  <h4>Table of Contents</h4>
  * this unordered seed list will be replaced by toc as unordered list
  {:toc}
</nav>

{% for project in site.data.openprojectlist %}
  {% if project.status == 'completed' %}
    {% continue %}
  {% endif %}

# {{ project.name }}

## Description

{{ project.description | markdownify }}

## Task ideas and expected results

{{ project.tasks | markdownify }}

{% endfor %}
