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
<div class="well" style="padding-left: 70px; padding-right: 70px" id={{meeting.label}}>
  <pubtit>{{ meeting.date }} at {{meeting.time_cest}} CEST</pubtit>
<div style="text-indent: 20px;">
  Connection information: {{meeting.connect}} <br />
 </div>
<div style="text-indent: 20px;">
  Agenda:
<ul style="margin-top:-10px;">
  {% for item in meeting.agenda %}
   <li> {{item.title}} (<em>{{item.speaker}} </em>): <a href="{{item.link.url}}">{{ item.link.display }}</a> </li>
  {% endfor %}
</ul>

 </div>
 </div>
{% endfor %} 
