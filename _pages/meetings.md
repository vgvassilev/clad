---
title: "Project Meetings"
layout: gridlay
excerpt: "Project Meetings"
sitemap: false
permalink: /meetings/
---



# Project Meetings
{% assign meetings_list = "" | split: ',' %}
{% for meeting_hash in site.data.meetings %}
{% for meeting in meeting_hash[1] %}
{% assign meetings_list = meetings_list | push: meeting %}
{% endfor %}
{% endfor %}

{% assign sorted_meetings = meetings_list | sort: "date" | reverse %}

{% for meeting in sorted_meetings %}
<div class="well" id={{meeting.label}}>
  <pubtit>{{ meeting.date }} at {{meeting.time_cest}} CEST</pubtit>
  Connection information: {{meeting.connect}} <br />
  Agenda:
  {% for item in meeting.agenda %}
   * {{item.title}} (<em>{{item.speaker}} </em>): [{{ item.link.display }}]({{ item.link.url }}) 
  {% endfor %}
 </div>
{% endfor %} 
